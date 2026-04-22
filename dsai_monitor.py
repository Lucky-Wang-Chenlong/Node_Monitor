#!/usr/bin/env python3
"""
Slurm multi-account GPU monitor

Features
- Monitor multiple users' running jobs:
  - allocated node(s)
  - allocated GPU count
  - per-node GPU metrics from nvidia-smi:
    memory used / total, utilization, power draw
- Monitor pending GPU jobs for those users
- Monitor partition idle rate / free GPU summary

Typical usage
-------------
python slurm_multi_account_monitor.py \
  --users zwang303 jchen293 \
  --partitions nvl h100 a100 \
  --interval 10

Optional JSON output
--------------------
python slurm_multi_account_monitor.py \
  --users zwang303 jchen293 \
  --partitions nvl h100 a100 \
  --json

Notes
-----
1. This script assumes:
   - you can run `squeue`, `sinfo`, `scontrol`
   - you can SSH to allocated nodes without interactive password prompts
   - nodes have `nvidia-smi`
2. GPU metrics are collected per node, not perfectly per job process. On a shared node,
   the GPU metrics shown are the whole-node GPU metrics. The script also reports how many
   GPUs the job requested / was allocated.
3. Partition idle rate is estimated from `sinfo` GRES / %G fields. On some clusters,
   GRES formatting differs. The parser below tries several common patterns.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Data models
# -----------------------------

@dataclass
class GPUMetric:
    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu_pct: int
    power_draw_w: float
    power_limit_w: float


@dataclass
class RunningJob:
    job_id: str
    user: str
    partition: str
    state: str
    reason: str
    nodes: List[str]
    node_count: int
    requested_gpus: Optional[int]
    elapsed: str
    name: str


@dataclass
class PendingJob:
    job_id: str
    user: str
    partition: str
    state: str
    reason: str
    requested_gpus: Optional[int]
    name: str


@dataclass
class PartitionStat:
    partition: str
    node_total: int
    node_idle: int
    node_alloc: int
    node_mix: int
    gpu_total: Optional[int]
    gpu_used_est: Optional[int]
    gpu_free_est: Optional[int]
    node_idle_rate: float
    gpu_idle_rate_est: Optional[float]


# -----------------------------
# Shell helpers
# -----------------------------


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
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}") from e



def run_ssh(node: str, remote_cmd: str, timeout: int = 20) -> str:
    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        node,
        remote_cmd,
    ]
    return run_cmd(ssh_cmd, timeout=timeout)


# -----------------------------
# Parsing helpers
# -----------------------------


def expand_nodelist(nodelist_expr: str) -> List[str]:
    nodelist_expr = nodelist_expr.strip()
    if not nodelist_expr:
        return []

    # Prefer scontrol if available; it correctly expands Slurm syntax.
    try:
        out = run_cmd(["scontrol", "show", "hostnames", nodelist_expr], timeout=10)
        nodes = [x.strip() for x in out.splitlines() if x.strip()]
        if nodes:
            return nodes
    except Exception:
        pass

    return [nodelist_expr]



def parse_gpu_count_from_tres(tres: str) -> Optional[int]:
    if not tres or tres == "N/A":
        return None
    m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres)
    if m:
        return int(m.group(1))
    return None



def parse_gpu_count_from_gres(gres: str) -> Optional[int]:
    if not gres or gres == "N/A":
        return None
    # Common examples:
    # gpu:4
    # gpu:h100:8
    # (null)
    m = re.search(r"gpu(?::[^:]+)?:(\d+)", gres)
    if m:
        return int(m.group(1))
    m = re.search(r"gpu:(\d+)", gres)
    if m:
        return int(m.group(1))
    return None



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


# -----------------------------
# Slurm queries
# -----------------------------


def get_running_jobs(users: List[str]) -> List[RunningJob]:
    user_expr = ",".join(users)
    fmt = "%i|%u|%P|%T|%R|%N|%D|%b|%M|%j"
    out = run_cmd([
        "squeue",
        "-u",
        user_expr,
        "-t",
        "RUNNING",
        "-h",
        "-o",
        fmt,
    ])

    jobs: List[RunningJob] = []
    for line in out.splitlines():
        parts = line.split("|", 9)
        if len(parts) != 10:
            continue
        job_id, user, partition, state, reason, nodelist, node_count, gres, elapsed, name = parts
        nodes = expand_nodelist(nodelist)
        jobs.append(
            RunningJob(
                job_id=job_id,
                user=user,
                partition=partition,
                state=state,
                reason=reason,
                nodes=nodes,
                node_count=safe_int(node_count, len(nodes)),
                requested_gpus=parse_gpu_count_from_gres(gres),
                elapsed=elapsed,
                name=name,
            )
        )
    return jobs



def get_pending_jobs(users: List[str]) -> List[PendingJob]:
    user_expr = ",".join(users)
    fmt = "%i|%u|%P|%T|%R|%b|%j"
    out = run_cmd([
        "squeue",
        "-u",
        user_expr,
        "-t",
        "PENDING",
        "-h",
        "-o",
        fmt,
    ])

    jobs: List[PendingJob] = []
    for line in out.splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7:
            continue
        job_id, user, partition, state, reason, gres, name = parts
        jobs.append(
            PendingJob(
                job_id=job_id,
                user=user,
                partition=partition,
                state=state,
                reason=reason,
                requested_gpus=parse_gpu_count_from_gres(gres),
                name=name,
            )
        )
    return jobs



def get_partition_stats(partitions: Optional[List[str]] = None) -> List[PartitionStat]:
    # We query node state summary first.
    fmt = "%P|%D|%t|%G|%C"
    args = ["sinfo", "-h", "-o", fmt]
    if partitions:
        args.extend(["-p", ",".join(partitions)])
    out = run_cmd(args)

    # Aggregate rows by partition, since sinfo often returns multiple rows by state / node feature.
    agg: Dict[str, Dict[str, object]] = {}

    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        partition, node_count_s, state, gres, cpu_state = parts
        partition = partition.rstrip("*")
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

        # State buckets
        if state_low.startswith("idle"):
            agg[partition]["node_idle"] = int(agg[partition]["node_idle"]) + node_count
        elif state_low.startswith("alloc"):
            agg[partition]["node_alloc"] = int(agg[partition]["node_alloc"]) + node_count
        elif state_low.startswith("mix"):
            agg[partition]["node_mix"] = int(agg[partition]["node_mix"]) + node_count

        # Try to parse total GPUs per node from GRES, then multiply by node count.
        per_node_gpu = parse_gpu_count_from_gres(gres)
        if per_node_gpu is not None:
            agg[partition]["gpu_total"] = int(agg[partition]["gpu_total"]) + per_node_gpu * node_count
            agg[partition]["gpu_seen"] = True

    # Estimate used GPUs via running jobs in those partitions.
    running_all = get_running_jobs_for_all_partitions(partitions)
    used_by_partition: Dict[str, int] = {}
    for job in running_all:
        if job.requested_gpus is not None:
            used_by_partition[job.partition] = used_by_partition.get(job.partition, 0) + job.requested_gpus

    stats: List[PartitionStat] = []
    for partition, info in sorted(agg.items()):
        node_total = int(info["node_total"])
        node_idle = int(info["node_idle"])
        node_alloc = int(info["node_alloc"])
        node_mix = int(info["node_mix"])
        gpu_total = int(info["gpu_total"]) if bool(info["gpu_seen"]) else None
        gpu_used = used_by_partition.get(partition)

        gpu_free = None
        gpu_idle_rate = None
        if gpu_total is not None:
            gpu_used = min(gpu_used, gpu_total) if gpu_used is not None else None
            gpu_free = gpu_total - gpu_used if gpu_used is not None else None
            gpu_idle_rate = (gpu_free / gpu_total) if (gpu_free is not None and gpu_total > 0) else None

        node_idle_rate = (node_idle / node_total) if node_total > 0 else 0.0
        stats.append(
            PartitionStat(
                partition=partition,
                node_total=node_total,
                node_idle=node_idle,
                node_alloc=node_alloc,
                node_mix=node_mix,
                gpu_total=gpu_total,
                gpu_used_est=gpu_used,
                gpu_free_est=gpu_free,
                node_idle_rate=node_idle_rate,
                gpu_idle_rate_est=gpu_idle_rate,
            )
        )

    return stats



def get_running_jobs_for_all_partitions(partitions: Optional[List[str]] = None) -> List[RunningJob]:
    fmt = "%i|%u|%P|%T|%R|%N|%D|%b|%M|%j"
    args = ["squeue", "-t", "RUNNING", "-h", "-o", fmt]
    if partitions:
        args.extend(["-p", ",".join(partitions)])
    out = run_cmd(args)

    jobs: List[RunningJob] = []
    for line in out.splitlines():
        parts = line.split("|", 9)
        if len(parts) != 10:
            continue
        job_id, user, partition, state, reason, nodelist, node_count, gres, elapsed, name = parts
        nodes = expand_nodelist(nodelist)
        jobs.append(
            RunningJob(
                job_id=job_id,
                user=user,
                partition=partition,
                state=state,
                reason=reason,
                nodes=nodes,
                node_count=safe_int(node_count, len(nodes)),
                requested_gpus=parse_gpu_count_from_gres(gres),
                elapsed=elapsed,
                name=name,
            )
        )
    return jobs


# -----------------------------
# GPU metrics
# -----------------------------


def get_gpu_metrics_for_node(node: str) -> List[GPUMetric]:
    remote_cmd = (
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit "
        "--format=csv,noheader,nounits"
    )
    out = run_ssh(node, remote_cmd, timeout=15)
    metrics: List[GPUMetric] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 7:
            continue
        idx, name, mem_used, mem_total, util, pdraw, plimit = parts
        metrics.append(
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
    return metrics


# -----------------------------
# Presentation helpers
# -----------------------------


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.1f}%"



def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)



def render_running_jobs(running_jobs: List[RunningJob]) -> None:
    print_header("RUNNING JOBS")
    if not running_jobs:
        print("No running jobs found.")
        return

    for job in running_jobs:
        print(
            f"[JOB {job.job_id}] user={job.user} partition={job.partition} name={job.name} "
            f"elapsed={job.elapsed} requested_gpus={job.requested_gpus if job.requested_gpus is not None else 'N/A'}"
        )
        print(f"  nodes: {', '.join(job.nodes) if job.nodes else 'N/A'}")
        for node in job.nodes:
            try:
                metrics = get_gpu_metrics_for_node(node)
                if not metrics:
                    print(f"  - {node}: no GPU metrics returned")
                    continue
                print(f"  - {node}:")
                for g in metrics:
                    print(
                        f"      GPU {g.index}: {g.memory_used_mb}/{g.memory_total_mb} MiB, "
                        f"util={g.utilization_gpu_pct}%, power={g.power_draw_w:.1f}/{g.power_limit_w:.1f} W"
                    )
            except Exception as e:
                print(f"  - {node}: failed to query GPU metrics ({e})")



def render_pending_jobs(pending_jobs: List[PendingJob]) -> None:
    print_header("PENDING GPU JOBS")
    if not pending_jobs:
        print("No pending jobs found.")
        return

    print(f"{'JOBID':<12} {'USER':<12} {'PARTITION':<12} {'GPUS':<6} {'STATE':<10} {'REASON':<28} NAME")
    for job in pending_jobs:
        gpus = str(job.requested_gpus) if job.requested_gpus is not None else "N/A"
        print(
            f"{job.job_id:<12} {job.user:<12} {job.partition:<12} {gpus:<6} {job.state:<10} {job.reason[:28]:<28} {job.name}"
        )



def render_partition_stats(stats: List[PartitionStat]) -> None:
    print_header("PARTITION IDLE / FREE CAPACITY")
    if not stats:
        print("No partition stats found.")
        return

    print(
        f"{'PARTITION':<12} {'NODES':<8} {'IDLE':<8} {'ALLOC':<8} {'MIX':<8} "
        f"{'NODE_IDLE%':<12} {'GPU_TOTAL':<10} {'GPU_USED~':<10} {'GPU_FREE~':<10} {'GPU_IDLE%~':<12}"
    )
    for s in stats:
        gpu_total = str(s.gpu_total) if s.gpu_total is not None else "N/A"
        gpu_used = str(s.gpu_used_est) if s.gpu_used_est is not None else "N/A"
        gpu_free = str(s.gpu_free_est) if s.gpu_free_est is not None else "N/A"
        print(
            f"{s.partition:<12} {s.node_total:<8} {s.node_idle:<8} {s.node_alloc:<8} {s.node_mix:<8} "
            f"{fmt_pct(s.node_idle_rate):<12} {gpu_total:<10} {gpu_used:<10} {gpu_free:<10} {fmt_pct(s.gpu_idle_rate_est):<12}"
        )



def render_json(running_jobs: List[RunningJob], pending_jobs: List[PendingJob], stats: List[PartitionStat]) -> None:
    gpu_cache: Dict[str, List[Dict[str, object]]] = {}
    for job in running_jobs:
        for node in job.nodes:
            if node in gpu_cache:
                continue
            try:
                gpu_cache[node] = [asdict(x) for x in get_gpu_metrics_for_node(node)]
            except Exception as e:
                gpu_cache[node] = [{"error": str(e)}]

    payload = {
        "running_jobs": [
            {
                **asdict(job),
                "node_gpu_metrics": {node: gpu_cache.get(node, []) for node in job.nodes},
            }
            for job in running_jobs
        ],
        "pending_jobs": [asdict(job) for job in pending_jobs],
        "partition_stats": [asdict(s) for s in stats],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# -----------------------------
# Main loop
# -----------------------------


def monitor_once(users: List[str], partitions: Optional[List[str]], as_json: bool) -> None:
    running_jobs = get_running_jobs(users)
    pending_jobs = get_pending_jobs(users)
    stats = get_partition_stats(partitions)

    if as_json:
        render_json(running_jobs, pending_jobs, stats)
    else:
        print(time.strftime("\n[%Y-%m-%d %H:%M:%S]"))
        render_running_jobs(running_jobs)
        render_pending_jobs(pending_jobs)
        render_partition_stats(stats)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Slurm jobs / GPU usage for multiple users.")
    parser.add_argument(
        "--users",
        nargs="+",
        required=True,
        help="One or more usernames to monitor, e.g. --users zwang303 jchen293",
    )
    parser.add_argument(
        "--partitions",
        nargs="*",
        default=None,
        help="Optional partitions to summarize, e.g. --partitions nvl h100 a100",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Polling interval in seconds. 0 means run once and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of pretty text.",
    )
    return parser.parse_args()



def main() -> int:
    try:
        require_cmd("squeue")
        require_cmd("sinfo")
        require_cmd("scontrol")
        require_cmd("ssh")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    args = parse_args()

    while True:
        try:
            monitor_once(args.users, args.partitions, args.json)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 0
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)

        if args.interval <= 0:
            break

        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
