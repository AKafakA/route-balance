"""One vLLM-SR stack per run (D146: a fresh deployment): new containers, a new CLI state directory, empty storage
volumes; torn down after the run once its evidence is copied out.

Every command is upstream's (`vllm-sr serve`, `vllm-sr stop`) or a read (`docker ps`, `docker inspect`, `docker logs`,
`docker cp` out of a container). The one input of ours is the Envoy template directory, passed as the CLI's
TEMPLATE_ROOT (D146, D150: the owner's timeouts and circuit breakers); its digest is recorded with every run.
The classifier weights the CLI downloads (`models/`) are shared by every run, like the engines' Hub snapshots: they
are read-only inputs, and their listing digest is recorded with every run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from routebalance.config import ConfigError
from vsrbench import constants as C

CLEAN_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"   # the CLI's own tools (docker) only
ACCESS_LOG = "/var/log/envoy_access.log"      # the CLI template's access-log path (in the Envoy container)
# Envoy writes its file access log on a timer (--file-flush-interval-msec, 10 s by default; the CLI does not set it):
# after a run the collector waits until every request of the run is in the log, bounded by two intervals (evidence
# collection only, after the last reply)
ENVOY_FLUSH_S = 10.0
STATE_FILES = ("runtime-config.yaml", "runtime-config.provenance.json", "envoy.yaml", "router-runtime.json")


_RID = re.compile(r'" "([^"]*)" "[^"]*" "[^"]*"$', re.M)     # the x-request-id field of an access-log line


class Docker:
    """docker commands, optionally through `sg docker` (a login session that predates the group membership)."""

    def __init__(self, sg: bool):
        self.sg = sg

    def argv(self, args: list[str]) -> list[str]:
        cmd = ["docker", *args]
        return ["sg", "docker", "-c", shlex.join(cmd)] if self.sg else cmd

    def run(self, args: list[str], *, check: bool = True, stdout=subprocess.PIPE, input: Optional[bytes] = None,
            merge_stderr: bool = False) -> bytes:
        proc = subprocess.run(self.argv(args), stdout=stdout, input=input,
                              stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE)
        if check and proc.returncode != 0:
            err = b"" if merge_stderr else proc.stderr
            raise ConfigError(f"docker {' '.join(args[:3])} failed ({proc.returncode}): "
                              f"{err.decode(errors='replace')[-800:]}")
        return proc.stdout if stdout is subprocess.PIPE else b""

    def names(self) -> list[str]:
        out = self.run(["ps", "-a", "--filter", "name=^vllm-sr", "--format", "{{.Names}}"]).decode()
        return sorted(n for n in out.split() if n)

    def volumes(self) -> list[str]:
        out = self.run(["volume", "ls", "--format", "{{.Name}}"]).decode()
        return sorted(v for v in out.split() if v in C.VOLUMES)

    def volumes_created(self) -> dict[str, str]:
        out = self.run(["volume", "inspect", *C.VOLUMES, "--format", "{{.Name}} {{.CreatedAt}}"]).decode()
        return dict(line.split(" ", 1) for line in out.splitlines() if line.strip())

    def exec_sh(self, container: str, script: str) -> str:
        return self.run(["exec", container, "sh", "-c", script]).decode(errors="replace").strip()


def tree_digest(root: Path) -> str:
    """sha256 over every file's relative path and bytes (the template directory)."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def listing_digest(root: Path) -> str:
    """sha256 over every file's relative path, size and mtime (the shared classifier weights, 8 GB: a listing)."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


class Stack:
    def __init__(self, workdir: Path, vsr: Path, template_root: Path, models_cache: Path, docker: Docker):
        self.workdir, self.vsr = Path(workdir), Path(vsr)
        self.template_root, self.models_cache = Path(template_root), Path(models_cache)
        self.docker = docker
        self.started = False       # set once this object launched a stack; only then does teardown touch one

    def _cli(self, args: list[str], log: Path) -> None:
        # a clean environment, as scripts/node_ctl.sh starts every lease process: upstream forwards host variables
        # (HF_*, SR_LOG_*, VLLM_SR_*) into its containers, so none may leak in from the operator's shell
        cmd = ["env", "-i", f"HOME={os.environ['HOME']}", f"PATH={CLEAN_PATH}", "LANG=C.UTF-8",
               f"TEMPLATE_ROOT={self.template_root.resolve()}", str(self.vsr), *args]
        if self.docker.sg:
            cmd = ["sg", "docker", "-c", shlex.join(cmd)]
        with open(log, "ab") as f:
            rc = subprocess.run(cmd, cwd=self.workdir, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise ConfigError(f"vllm-sr {args[0]} exited {rc} (see {log})")

    def clean(self) -> list[str]:
        """What a previous stack left: containers or storage volumes (a fresh stack starts from none)."""
        return [f"container {n}" for n in self.docker.names()] + [f"volume {v}" for v in self.docker.volumes()]

    def up(self, config: Path) -> dict:
        left = self.clean()
        if left:
            raise ConfigError(f"a previous vLLM-SR stack is still present ({left}): tear it down first")
        if self.workdir.exists():
            raise ConfigError(f"{self.workdir} exists: every run gets a new CLI state directory")
        if not (self.models_cache / ".").is_dir():
            raise ConfigError(f"no classifier weights at {self.models_cache}")
        self.workdir.mkdir(parents=True)
        self.started = True
        (self.workdir / "config.yaml").write_bytes(Path(config).read_bytes())
        (self.workdir / "models").symlink_to(self.models_cache.resolve(), target_is_directory=True)
        from vsrbench.run_cell import listing        # imported here: run_cell imports this module
        record = {"template_root": str(self.template_root), "template_sha256": tree_digest(self.template_root),
                  "models_listing_sha256": listing_digest(self.models_cache), "models_listing": listing(self.models_cache),
                  "config_sha256": hashlib.sha256((self.workdir / "config.yaml").read_bytes()).hexdigest()}
        self._cli(["serve", "--config", "config.yaml", "--minimal", "--replace-active-config"],
                  self.workdir / "serve.log")
        return record

    def inspect(self) -> list[dict]:
        """Each container as Docker runs it, without its environment (the storage passwords live there)."""
        names = self.docker.names()
        if not names:
            return []
        raw = json.loads(self.docker.run(["inspect", *names]))
        keep = []
        for c in raw:
            image = json.loads(self.docker.run(["image", "inspect", c["Image"]]))[0]
            keep.append({"name": c["Name"].lstrip("/"), "image": c["Config"]["Image"], "image_id": c["Image"],
                         "repo_digests": image.get("RepoDigests", []), "running": c["State"]["Running"],
                         "started_at": c["State"]["StartedAt"], "restart_count": c["RestartCount"],
                         "args": c["Args"][-4:], "mounts": [(m["Source"], m["Destination"]) for m in c["Mounts"]],
                         "ports": c["HostConfig"]["PortBindings"],
                         "env": {k: ("<secret>" if any(w in k.upper() for w in ("PASS", "SECRET", "TOKEN", "KEY"))
                                     else v) for k, v in (e.split("=", 1) for e in c["Config"]["Env"] or [])},
                         "resources": {k: c["HostConfig"].get(k) for k in ("DeviceRequests", "Devices", "Privileged",
                                                                           "NanoCpus", "CpusetCpus", "Memory", "Runtime")}})
        return keep

    def access_log_ids(self) -> set[str]:
        text = self.docker.run(["exec", "vllm-sr-envoy-container", "cat", ACCESS_LOG]).decode(errors="replace")
        return {m.group(1) for m in _RID.finditer(text)}

    def wait_access_log(self, ids: set[str]) -> dict:
        """Wait until Envoy has flushed a line for every request id of the run (at most 2 flush intervals)."""
        start = time.monotonic()
        while True:
            missing = ids - self.access_log_ids()
            waited = time.monotonic() - start
            if not missing or waited > 2 * ENVOY_FLUSH_S:
                return {"waited_s": round(waited, 3), "missing": sorted(missing)}
            time.sleep(1.0)

    def wait_replays(self, total_of, expected: int) -> dict:
        """Wait until the router's replay store holds the run's records (written as each stream completes), bounded as the
        access-log wait (evidence collection only, after the last reply)."""
        start = time.monotonic()
        while True:
            total = total_of()
            waited = time.monotonic() - start
            if total == expected or waited > 2 * ENVOY_FLUSH_S:
                return {"waited_s": round(waited, 3), "total": total, "expected": expected}
            time.sleep(1.0)

    def envoy_address(self) -> str:
        """Envoy's address on the stack's network, as Docker reports it (the path that bypasses docker-proxy)."""
        nets = json.loads(self.docker.run(["inspect", "vllm-sr-envoy-container", "--format",
                                           "{{json .NetworkSettings.Networks}}"]))
        return nets[C.STACK_NETWORK]["IPAddress"]

    def store_state(self) -> dict:
        """The stack's stored state, read inside its storage containers (no secret leaves them): every Postgres table's
        row count and the number of Redis keys."""
        psql = 'psql -U "$POSTGRES_USER" -d "${POSTGRES_DB:-$POSTGRES_USER}" -Atc '
        tables = self.docker.exec_sh("vllm-sr-postgres", psql + "\"select schemaname || '.' || relname "
                                     "from pg_stat_user_tables order by 1\"").split()
        counts = {t: int(self.docker.exec_sh("vllm-sr-postgres", psql + f"\"select count(*) from {t}\""))
                  for t in tables}
        keys = self.docker.exec_sh("vllm-sr-redis", 'redis-cli --no-auth-warning -a "$(sed -n "s/^requirepass //p" '
                            '/usr/local/etc/redis/redis.conf)" DBSIZE')
        return {"postgres_rows": counts, "redis_keys": int(keys)}

    def collect(self, out: Path) -> None:
        """Copy the run's evidence out before teardown: every container's full log stream (the router's host log
        file is a bounded spool; its stdout is not), Envoy's access log, the CLI's rendered state."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        for name in self.docker.names():
            with open(out / f"{name}.log", "wb") as f:
                self.docker.run(["logs", name], stdout=f, merge_stderr=True)   # the container's stdout and stderr
        self.docker.run(["cp", f"vllm-sr-envoy-container:{ACCESS_LOG}", str(out / "envoy_access.log")])
        state = self.workdir / ".vllm-sr"
        for n in STATE_FILES:
            src = state / n
            if src.is_file() and os.access(src, os.R_OK):
                (out / n).write_bytes(src.read_bytes())
        (out / "containers.json").write_text(json.dumps(self.inspect(), indent=1) + "\n")
        for n in ("serve.log", "stop.log"):
            if (self.workdir / n).is_file():
                (out / n).write_bytes((self.workdir / n).read_bytes())

    def down(self) -> None:
        if not self.started:
            return                  # a stack this object did not launch (a refused up) is never torn down from here
        self._cli(["stop"], self.workdir / "stop.log")
        for v in self.docker.volumes():
            self.docker.run(["volume", "rm", v])
        left = self.clean()
        if left:
            raise ConfigError(f"the stack did not tear down: {left}")
