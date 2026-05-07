#!/usr/bin/env python3
"""
Tag latency data with instance_type derived from instance_id and deployment config.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.tag_latency_instance_type \
        --input data/route_balance/latency_data/all/latency_train.jsonl \
        --config route_balance/config/route_balance/model_deployment.json \
        --output data/route_balance/latency_data/all/latency_train_tagged.jsonl
"""

import argparse
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# GPU type derived from CloudLab node type prefix
NODE_GPU_MAP = {
    "d8545": "a100",
    "c4130": "v100",
    "d7525": "a30",
    "c240g5": "p100",
}


def build_host_to_model_map(config_path: str) -> dict:
    """Build mapping from hostname to (model_short, gpu_type)."""
    with open(config_path) as f:
        config = json.load(f)

    host_map = {}
    for model_key, model_cfg in config.items():
        hf_name = model_cfg.get("hf_model_name", model_key)
        # e.g. "Qwen/Qwen2.5-72B" -> "qwen2.5-72b"
        model_short = hf_name.split("/")[-1].lower()
        for host_entry in model_cfg.get("node_hosts", []):
            # "anon@d8545-10s10301.cluster.example" -> "d8545-10s10301"
            hostname = host_entry.split("@")[-1].split(".")[0]
            node_prefix = hostname.split("-")[0]
            gpu_type = NODE_GPU_MAP.get(node_prefix, "unknown")
            instance_type = f"{model_short}_{gpu_type}"
            host_map[hostname] = instance_type

    return host_map


def main():
    parser = argparse.ArgumentParser(description="Tag latency data with instance_type")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--config", default="route_balance/config/route_balance/model_deployment.json")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()

    host_map = build_host_to_model_map(args.config)
    logger.info(f"Host map: {len(host_map)} entries")
    for host, itype in sorted(host_map.items()):
        logger.info(f"  {host} -> {itype}")

    counts = Counter()
    unmatched = Counter()
    total = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1

            inst_id = record.get("instance_id", "")
            # Extract hostname: "d8545-10s10301.cluster.example_port8300" -> "d8545-10s10301"
            hostname = inst_id.split(".")[0] if inst_id else ""

            instance_type = host_map.get(hostname)
            if instance_type:
                record["instance_type"] = instance_type
                counts[instance_type] += 1
            else:
                unmatched[hostname] += 1
                # Try prefix-based fallback
                node_prefix = hostname.split("-")[0] if hostname else ""
                gpu_type = NODE_GPU_MAP.get(node_prefix, "unknown")
                record["instance_type"] = f"unknown_{gpu_type}"
                counts[f"unknown_{gpu_type}"] += 1

            fout.write(json.dumps(record) + "\n")

    logger.info(f"Tagged {total} records:")
    for itype, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {itype}: {count}")
    if unmatched:
        logger.warning(f"Unmatched hostnames: {dict(unmatched)}")


if __name__ == "__main__":
    main()
