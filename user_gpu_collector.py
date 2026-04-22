#!/usr/bin/env python3
"""
Per-user Slurm collector.

Run this script under each user (ideally in tmux). It writes a JSON snapshot
that the center web program can aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib import request


def require_cmd(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required command not found in PATH: {cmd}")


def run_cmd(cmd: List[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=True,
        )
        return p.stdout
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or f"{cmd[0]} exit={exc.returncode}") from exc


def run_ssh(node: str, remote_cmd: str, timeout: int = 20) -> str:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=5",
        node,
        remote_cmd,
    ]
    return run_cmd(cmd, timeout=timeout)


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_gpu_count_from_gres(gres: str) -> Optional[int]:
    if not gres or gres == "N/A":
        return None
    m = re.search(r"gpu(?::[^:]+)?:(\d+)", gres)
    if m:
        return int(m.group(1))
    m = re.search(r"gpu:(\d+)", gres)
    if m:
        return int(m.group(1))
    return None


def parse_partition_candidates(partition_field: str) -> List[str]:
    if not partition_field:
        return []
    return [x.strip().rstrip("*") for x in partition_field.strip().rstrip("*").split(",") if x.strip()]


def expand_nodelist(nodelist_expr: str) -> List[str]:
    if not nodelist_expr or nodelist_expr == "N/A":
        return []
    out = run_cmd(["scontrol", "show", "hostnames", nodelist_expr], timeout=10)
    return [x.strip() for x in out.splitlines() if x.strip()]


def get_running_jobs(user: str) -> List[Dict[str, object]]:
    fmt = "%i|%u|%P|%T|%R|%N|%D|%b|%M|%j"
    out = run_cmd(["squeue", "-u", user, "-t", "RUNNING", "-h", "-o", fmt])
    jobs: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = line.split("|", 9)
        if len(parts) != 10:
            continue
        job_id, user_s, partition, state, reason, nodelist, node_count, gres, elapsed, name = parts
        partitions = parse_partition_candidates(partition)
        jobs.append(
            {
                "job_id": job_id,
                "user": user_s,
                "partition": partitions[0] if partitions else "",
                "partition_raw": partition.strip(),
                "partitions": partitions,
                "state": state,
                "reason": reason,
                "nodes": expand_nodelist(nodelist),
                "node_count": safe_int(node_count, 0),
                "requested_gpus": parse_gpu_count_from_gres(gres),
                "elapsed": elapsed,
                "name": name,
            }
        )
    return jobs


def get_pending_jobs(user: str) -> List[Dict[str, object]]:
    fmt = "%i|%u|%P|%T|%R|%b|%j"
    out = run_cmd(["squeue", "-u", user, "-t", "PENDING", "-h", "-o", fmt])
    jobs: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7:
            continue
        job_id, user_s, partition, state, reason, gres, name = parts
        partitions = parse_partition_candidates(partition)
        jobs.append(
            {
                "job_id": job_id,
                "user": user_s,
                "partition": partitions[0] if partitions else "",
                "partition_raw": partition.strip(),
                "partitions": partitions,
                "state": state,
                "reason": reason,
                "requested_gpus": parse_gpu_count_from_gres(gres),
                "name": name,
            }
        )
    return jobs


def get_node_gpu_metrics(node: str) -> List[Dict[str, object]]:
    cmd = (
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit "
        "--format=csv,noheader,nounits"
    )
    out = run_ssh(node, cmd, timeout=15)
    metrics: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 7:
            continue
        idx, name, mem_used, mem_total, util, pdraw, plimit = parts
        metrics.append(
            {
                "index": safe_int(idx),
                "name": name,
                "memory_used_mb": safe_int(mem_used),
                "memory_total_mb": safe_int(mem_total),
                "utilization_gpu_pct": safe_int(util),
                "power_draw_w": safe_float(pdraw),
                "power_limit_w": safe_float(plimit),
            }
        )
    return metrics


def collect_once(user: str) -> Dict[str, object]:
    running = get_running_jobs(user)
    pending = get_pending_jobs(user)
    node_cache: Dict[str, List[Dict[str, object]]] = {}

    for job in running:
        for node in job.get("nodes", []):
            node_s = str(node)
            if node_s in node_cache:
                continue
            try:
                node_cache[node_s] = get_node_gpu_metrics(node_s)
            except Exception as exc:
                node_cache[node_s] = [{"error": str(exc)}]

    for job in running:
        job["node_gpu_metrics"] = {str(node): node_cache.get(str(node), []) for node in job.get("nodes", [])}

    return {
        "user": user,
        "host": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "running_jobs": running,
        "pending_jobs": pending,
    }


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tf:
        json.dump(payload, tf, indent=2, ensure_ascii=False)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def push_to_center(push_url: str, payload: Dict[str, object], token: Optional[str], timeout: int = 8) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(push_url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if token:
        req.add_header("X-Collector-Token", token)
    with request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"center push failed: HTTP {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-user GPU collector for center monitor.")
    parser.add_argument("--user", required=True, help="User to collect.")
    parser.add_argument("--output-dir", default="./collector_data", help="Directory to write snapshots.")
    parser.add_argument("--push-url", default=None, help="Center endpoint, e.g. http://center:18080/api/collector")
    parser.add_argument("--push-token", default=None, help="Optional collector token for center auth.")
    parser.add_argument("--interval", type=int, default=10, help="Collect interval seconds.")
    return parser.parse_args()


def main() -> int:
    for cmd in ["squeue", "scontrol", "ssh"]:
        require_cmd(cmd)

    args = parse_args()
    out_file = Path(args.output_dir).resolve() / f"{args.user}.json"
    interval = max(args.interval, 3)
    print(f"[collector] user={args.user} output={out_file} interval={interval}s")
    if args.push_url:
        print(f"[collector] push_url={args.push_url}")

    while True:
        try:
            payload = collect_once(args.user)
            write_json_atomic(out_file, payload)
            if args.push_url:
                push_to_center(args.push_url, payload, args.push_token)
        except KeyboardInterrupt:
            print("[collector] stopped")
            return 0
        except Exception as exc:
            err_payload = {
                "user": args.user,
                "host": socket.gethostname(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(exc),
                "running_jobs": [],
                "pending_jobs": [],
            }
            write_json_atomic(out_file, err_payload)
            if args.push_url:
                try:
                    push_to_center(args.push_url, err_payload, args.push_token)
                except Exception as push_exc:
                    print(f"[collector] push error: {push_exc}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
