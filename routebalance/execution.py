"""What the scheduler process executes, read from the process itself for the manifest and the statistics: the code's
digest, every installed distribution, the native thread pools, the open-file limit, the kernel's listen-queue
counters and the HTTP server's settings. Recorded only, never a reason to refuse anything."""
from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import resource
import sys
from pathlib import Path
from typing import Optional

import aiohttp
import uvicorn

from routebalance.config import ConfigError

log = logging.getLogger(__name__)

MANIFEST_FORMAT = "rb-manifest/1"
PACKAGE_DIR = Path(__file__).resolve().parent
# Environment variables that size native thread pools (OpenMP, BLAS, the tokenizers' Rayon pool): recorded,
# never read as settings.
THREAD_ENV = ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY", "MKL_NUM_THREADS",
              "MKL_DYNAMIC", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
              "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES")


def py_tree_digest(directory: Path) -> str:
    """sha256 of the sha256sum listing of every .py file under `directory`, by path relative to it, in byte
    order: `(cd DIR && find . -name '*.py' -printf '%P\\n' | LC_ALL=C sort | xargs sha256sum) | sha256sum`."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError(f"{directory} is not a directory")
    files = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*.py") if p.is_file())
    listing = "".join(f"{hashlib.sha256((directory / f).read_bytes()).hexdigest()}  {f}\n" for f in files)
    return hashlib.sha256(listing.encode()).hexdigest()


def library_versions() -> dict[str, str]:
    """The interpreter and every installed distribution (not a chosen list: a dependency of a dependency, such
    as aiohttp's connection library, runs too). Two installs of one name are joined, so they show."""
    found: dict[str, list[str]] = {}
    for dist in importlib.metadata.distributions():
        name = str(dist.metadata["Name"]).lower().replace("_", "-")
        found.setdefault(name, []).append(dist.version)   # default-ok: grouping
    return {"python": sys.version, **{n: "|".join(sorted(v)) for n, v in sorted(found.items())}}


def native_threadpools() -> dict[str, int]:
    """{"<api>:<library dir>/<library file>": threads} for every native thread pool loaded in the process
    (threadpoolctl), as a thread that never set its own count sees it: the launch environment's value."""
    from threadpoolctl import threadpool_info      # installed with scikit-learn
    out: dict[str, int] = {}
    for pool in threadpool_info():
        lib = Path(pool["filepath"])
        key = base = f"{pool['internal_api']}:{lib.parent.name}/{lib.name}"
        n = 2
        while key in out:
            key, n = f"{base}#{n}", n + 1
        out[key] = pool["num_threads"]
    return out


NOFILE_SOFT = 65535          # v3's open-file target (its set_ulimit, as vLLM's)


def set_ulimit(target_soft_limit: int = NOFILE_SOFT) -> None:
    """v3's set_ulimit, as vLLM's: the soft open-file
    limit raised to the target when it is lower, the hard limit unchanged; a refusal is logged as v3 logged it, and
    the manifest records the limit the process runs with."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < target_soft_limit:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft_limit, hard))
        except ValueError as e:
            log.warning("Found ulimit of %s and failed to automatically increase with error %s. This can cause fd "
                        "limit errors like `OSError: [Errno 24] Too many open files`. Consider increasing with "
                        "ulimit -n", soft, e)


def open_files_limit() -> dict[str, int]:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return {"soft": soft, "hard": hard}


def tcp_listen_counters() -> dict[str, int]:
    """The node's kernel counters of connections a full listen queue turned away (TcpExt ListenOverflows and
    ListenDrops in /proc/net/netstat), cumulative since boot; stats_before/after bracket a run. Recorded only."""
    lines = Path("/proc/net/netstat").read_text().splitlines()
    for head, values in zip(lines[::2], lines[1::2], strict=True):
        if head.startswith("TcpExt:"):
            row = dict(zip(head.split()[1:], values.split()[1:], strict=True))
            return {"overflows": int(row["ListenOverflows"]), "drops": int(row["ListenDrops"])}
    raise RuntimeError("/proc/net/netstat has no TcpExt counters")


def http_settings(server: Optional[uvicorn.Server]) -> Optional[dict]:
    """What the listening server executes, read from its loaded configuration, and the kernel's cap on the listen
    backlog (the queue a connection waits in until the event loop accepts it)."""
    if server is None or not server.config.loaded:
        return None
    c, http = server.config, server.config.http_protocol_class
    return {"server": f"uvicorn {uvicorn.__version__}", "http": f"{http.__module__}.{http.__qualname__}",
            "backlog": c.backlog, "somaxconn": int(Path("/proc/sys/net/core/somaxconn").read_text()),
            "timeout_keep_alive": c.timeout_keep_alive, "access_log": c.access_log,
            "limit_concurrency": c.limit_concurrency, "workers": c.workers}


def session_settings(session: aiohttp.ClientSession) -> dict:
    conn = session.connector
    return {"limit": conn.limit, "limit_per_host": conn.limit_per_host, "force_close": conn.force_close}


def type_name(obj) -> Optional[str]:
    return None if obj is None else type(obj).__name__
