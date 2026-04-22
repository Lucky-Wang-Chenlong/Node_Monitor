#!/usr/bin/env python3
"""
Slurm multi-account monitor with web dashboard API.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class GPUMetric:
    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu_pct: int
    power_draw_w: float
    power_limit_w: float


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
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or f"exit code {exc.returncode}"
        raise RuntimeError(f"{cmd[0]} failed: {details}") from exc


def run_ssh(node: str, remote_cmd: str, timeout: int = 20) -> str:
    ssh_cmd = [
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
    try:
        return run_cmd(ssh_cmd, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"cannot query node {node} via ssh ({exc})") from exc


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
    cleaned = partition_field.strip().rstrip("*")
    parts = [x.strip().rstrip("*") for x in cleaned.split(",") if x.strip()]
    return parts


def normalize_partition(partition: str) -> str:
    candidates = parse_partition_candidates(partition)
    return candidates[0] if candidates else ""


def expand_nodelist(nodelist_expr: str) -> List[str]:
    if not nodelist_expr or nodelist_expr == "N/A":
        return []
    out = run_cmd(["scontrol", "show", "hostnames", nodelist_expr], timeout=10)
    return [x.strip() for x in out.splitlines() if x.strip()]


def get_running_jobs(users: List[str]) -> List[Dict[str, object]]:
    user_expr = ",".join(users)
    fmt = "%i|%u|%P|%T|%R|%N|%D|%b|%M|%j"
    out = run_cmd(
        ["squeue", "-u", user_expr, "-t", "RUNNING", "-h", "-o", fmt]
    )

    jobs: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = line.split("|", 9)
        if len(parts) != 10:
            continue
        job_id, user, partition, state, reason, nodelist, node_count, gres, elapsed, name = parts
        partition_candidates = parse_partition_candidates(partition)
        jobs.append(
            {
                "job_id": job_id,
                "user": user,
                "partition": partition_candidates[0] if partition_candidates else normalize_partition(partition),
                "partition_raw": partition.strip(),
                "partitions": partition_candidates,
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


def get_pending_jobs(users: List[str]) -> List[Dict[str, object]]:
    user_expr = ",".join(users)
    fmt = "%i|%u|%P|%T|%R|%b|%j"
    out = run_cmd(
        ["squeue", "-u", user_expr, "-t", "PENDING", "-h", "-o", fmt]
    )

    jobs: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7:
            continue
        job_id, user, partition, state, reason, gres, name = parts
        partition_candidates = parse_partition_candidates(partition)
        jobs.append(
            {
                "job_id": job_id,
                "user": user,
                "partition": partition_candidates[0] if partition_candidates else normalize_partition(partition),
                "partition_raw": partition.strip(),
                "partitions": partition_candidates,
                "state": state,
                "reason": reason,
                "requested_gpus": parse_gpu_count_from_gres(gres),
                "name": name,
            }
        )
    return jobs


def get_running_jobs_all_partitions(partitions: Optional[List[str]]) -> List[Dict[str, object]]:
    fmt = "%i|%u|%P|%T|%R|%N|%D|%b|%M|%j"
    args = ["squeue", "-t", "RUNNING", "-h", "-o", fmt]
    if partitions:
        args.extend(["-p", ",".join(partitions)])
    out = run_cmd(args)

    jobs: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = line.split("|", 9)
        if len(parts) != 10:
            continue
        job_id, user, partition, state, reason, nodelist, node_count, gres, elapsed, name = parts
        partition_candidates = parse_partition_candidates(partition)
        jobs.append(
            {
                "job_id": job_id,
                "user": user,
                "partition": partition_candidates[0] if partition_candidates else normalize_partition(partition),
                "partition_raw": partition.strip(),
                "partitions": partition_candidates,
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


def get_partition_stats(partitions: Optional[List[str]]) -> List[Dict[str, object]]:
    fmt = "%P|%D|%t|%G"
    args = ["sinfo", "-h", "-o", fmt]
    if partitions:
        args.extend(["-p", ",".join(partitions)])
    out = run_cmd(args)

    agg: Dict[str, Dict[str, object]] = {}
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        partition, node_count_s, state, gres = parts
        partition = normalize_partition(partition)
        node_count = safe_int(node_count_s, 0)
        state_low = state.lower()

        if partition not in agg:
            agg[partition] = {
                "node_total": 0,
                "node_idle": 0,
                "node_alloc": 0,
                "node_mix": 0,
                "gpu_total": 0,
                "gpu_seen": False,
            }

        agg[partition]["node_total"] = int(agg[partition]["node_total"]) + node_count
        if state_low.startswith("idle"):
            agg[partition]["node_idle"] = int(agg[partition]["node_idle"]) + node_count
        elif state_low.startswith("alloc"):
            agg[partition]["node_alloc"] = int(agg[partition]["node_alloc"]) + node_count
        elif state_low.startswith("mix"):
            agg[partition]["node_mix"] = int(agg[partition]["node_mix"]) + node_count

        per_node_gpu = parse_gpu_count_from_gres(gres)
        if per_node_gpu is not None:
            agg[partition]["gpu_total"] = int(agg[partition]["gpu_total"]) + per_node_gpu * node_count
            agg[partition]["gpu_seen"] = True

    used_by_partition: Dict[str, int] = {}
    for job in get_running_jobs_all_partitions(partitions):
        req = job.get("requested_gpus")
        part = str(job.get("partition", ""))
        if isinstance(req, int):
            used_by_partition[part] = used_by_partition.get(part, 0) + req

    stats: List[Dict[str, object]] = []
    for partition, info in sorted(agg.items()):
        node_total = int(info["node_total"])
        node_idle = int(info["node_idle"])
        node_alloc = int(info["node_alloc"])
        node_mix = int(info["node_mix"])
        gpu_total = int(info["gpu_total"]) if bool(info["gpu_seen"]) else None
        gpu_used = used_by_partition.get(partition)

        gpu_free = None
        gpu_idle_rate = None
        if gpu_total is not None and gpu_total > 0 and gpu_used is not None:
            gpu_used = min(gpu_used, gpu_total)
            gpu_free = gpu_total - gpu_used
            gpu_idle_rate = gpu_free / gpu_total

        node_idle_rate = (node_idle / node_total) if node_total > 0 else 0.0
        stats.append(
            {
                "partition": partition,
                "node_total": node_total,
                "node_idle": node_idle,
                "node_alloc": node_alloc,
                "node_mix": node_mix,
                "node_idle_rate": node_idle_rate,
                "gpu_total": gpu_total,
                "gpu_used_est": gpu_used,
                "gpu_free_est": gpu_free,
                "gpu_idle_rate_est": gpu_idle_rate,
            }
        )
    return stats


def get_gpu_metrics_for_node(node: str) -> List[Dict[str, object]]:
    remote_cmd = (
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit "
        "--format=csv,noheader,nounits"
    )
    out = run_ssh(node, remote_cmd, timeout=15)
    metrics: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 7:
            continue
        idx, name, mem_used, mem_total, util, pdraw, plimit = parts
        metrics.append(
            asdict(
                GPUMetric(
                    index=safe_int(idx),
                    name=name,
                    memory_used_mb=safe_int(mem_used),
                    memory_total_mb=safe_int(mem_total),
                    utilization_gpu_pct=safe_int(util),
                    power_draw_w=safe_float(pdraw),
                    power_limit_w=safe_float(plimit),
                )
            )
        )
    return metrics


def collect_snapshot(users: List[str], partitions: Optional[List[str]]) -> Dict[str, object]:
    running_jobs = get_running_jobs(users)
    pending_jobs = get_pending_jobs(users)
    partition_stats = get_partition_stats(partitions)
    part_index = {x["partition"]: x for x in partition_stats}

    node_gpu_cache: Dict[str, List[Dict[str, object]]] = {}
    for job in running_jobs:
        for node in job.get("nodes", []):
            node_s = str(node)
            if node_s in node_gpu_cache:
                continue
            try:
                node_gpu_cache[node_s] = get_gpu_metrics_for_node(node_s)
            except Exception as exc:
                node_gpu_cache[node_s] = [{"error": str(exc)}]

    for job in running_jobs:
        metrics_by_node: Dict[str, List[Dict[str, object]]] = {}
        for node in job.get("nodes", []):
            node_s = str(node)
            metrics_by_node[node_s] = node_gpu_cache.get(node_s, [])
        job["node_gpu_metrics"] = metrics_by_node

    pending_by_partition: Dict[str, Dict[str, object]] = {}
    for job in pending_jobs:
        targets = job.get("partitions")
        if not isinstance(targets, list) or not targets:
            targets = [str(job.get("partition", "UNKNOWN"))]
        for partition in targets:
            partition = str(partition) if partition else "UNKNOWN"
            if partition not in pending_by_partition:
                st = part_index.get(partition, {})
                pending_by_partition[partition] = {
                    "partition": partition,
                    "pending_jobs": 0,
                    "pending_gpus": 0,
                    "node_idle_rate": st.get("node_idle_rate"),
                    "gpu_idle_rate_est": st.get("gpu_idle_rate_est"),
                    "gpu_free_est": st.get("gpu_free_est"),
                }
            pending_by_partition[partition]["pending_jobs"] = int(pending_by_partition[partition]["pending_jobs"]) + 1
            req = job.get("requested_gpus")
            if isinstance(req, int):
                pending_by_partition[partition]["pending_gpus"] = int(pending_by_partition[partition]["pending_gpus"]) + req

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "users": users,
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "partition_stats": partition_stats,
        "pending_by_partition": sorted(pending_by_partition.values(), key=lambda x: str(x["partition"])),
    }


class SnapshotCache:
    def __init__(self, users: List[str], partitions: Optional[List[str]], refresh_seconds: int):
        self.users = users
        self.partitions = partitions
        self.refresh_seconds = refresh_seconds
        self._last_ts = 0.0
        self._snapshot: Dict[str, object] = {
            "timestamp": None,
            "error": "No snapshot yet.",
        }

    def get_snapshot(self) -> Dict[str, object]:
        now = time.time()
        if now - self._last_ts >= self.refresh_seconds:
            try:
                self._snapshot = collect_snapshot(self.users, self.partitions)
            except Exception as exc:
                self._snapshot = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(exc),
                }
            self._last_ts = now
        return self._snapshot


class MonitorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, cache: SnapshotCache, web_root: str, **kwargs):
        self.cache = cache
        super().__init__(*args, directory=web_root, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            payload = self.cache.get_snapshot()
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slurm web monitor for multi-account GPU jobs.")
    parser.add_argument("--users", nargs="+", required=True, help="Users to monitor.")
    parser.add_argument("--partitions", nargs="*", default=None, help="Optional partitions to summarize.")
    parser.add_argument("--refresh", type=int, default=15, help="API refresh interval in seconds.")
    parser.add_argument("--serve", action="store_true", help="Start web server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=18080, help="Bind port.")
    return parser.parse_args()


def main() -> int:
    for cmd in ["squeue", "sinfo", "scontrol", "ssh"]:
        require_cmd(cmd)

    args = parse_args()
    if not args.serve:
        print(json.dumps(collect_snapshot(args.users, args.partitions), indent=2, ensure_ascii=False))
        return 0

    base_dir = Path(__file__).resolve().parent
    web_root = str(base_dir / "web")
    cache = SnapshotCache(args.users, args.partitions, max(args.refresh, 3))

    def handler(*handler_args, **handler_kwargs):
        return MonitorHandler(*handler_args, cache=cache, web_root=web_root, **handler_kwargs)

    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
