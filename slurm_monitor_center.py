#!/usr/bin/env python3
"""
Center monitor: aggregate per-user collector JSONs and serve web dashboard.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse


def require_cmd(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required command not found in PATH: {cmd}")


def run_cmd(cmd: List[str], timeout: int = 20) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=True,
    )
    return p.stdout


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
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


def normalize_partition(partition: str) -> str:
    parts = parse_partition_candidates(partition)
    return parts[0] if parts else ""


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
        _ = nodelist
        jobs.append(
            {
                "job_id": job_id,
                "user": user,
                "partition": normalize_partition(partition),
                "state": state,
                "reason": reason,
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
        stats.append(
            {
                "partition": partition,
                "node_total": node_total,
                "node_idle": node_idle,
                "node_alloc": node_alloc,
                "node_mix": node_mix,
                "node_idle_rate": (node_idle / node_total) if node_total > 0 else 0.0,
                "gpu_total": gpu_total,
                "gpu_used_est": gpu_used,
                "gpu_free_est": gpu_free,
                "gpu_idle_rate_est": gpu_idle_rate,
            }
        )
    return stats


def read_user_snapshot(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"error": "invalid payload format"}
        return payload
    except FileNotFoundError:
        return {"error": f"collector file not found: {path.name}"}
    except Exception as exc:
        return {"error": str(exc)}


class CollectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_user: Dict[str, Dict[str, object]] = {}

    def update(self, user: str, payload: Dict[str, object]) -> None:
        with self._lock:
            self._by_user[user] = {
                "payload": payload,
                "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    def get(self, user: str) -> Optional[Dict[str, object]]:
        with self._lock:
            return self._by_user.get(user)


class GpuHistoryStore:
    def __init__(self, max_points: int = 360) -> None:
        self._lock = threading.Lock()
        self._max_points = max_points
        self._series: Dict[str, List[Dict[str, object]]] = {}

    @staticmethod
    def _key(user: str, node: str, gpu_index: int) -> str:
        return f"{user}|{node}|{gpu_index}"

    def append_from_snapshot(self, running_jobs: List[Dict[str, object]], ts_epoch: float) -> None:
        with self._lock:
            for job in running_jobs:
                user = str(job.get("user", ""))
                node_gpu_metrics = job.get("node_gpu_metrics", {})
                if not user or not isinstance(node_gpu_metrics, dict):
                    continue
                for node, gpu_list in node_gpu_metrics.items():
                    if not isinstance(gpu_list, list):
                        continue
                    for gpu in gpu_list:
                        if not isinstance(gpu, dict) or gpu.get("error") is not None:
                            continue
                        index = gpu.get("index")
                        if not isinstance(index, int):
                            continue
                        key = self._key(user, str(node), index)
                        point = {
                            "ts": int(ts_epoch),
                            "memory_used_mb": gpu.get("memory_used_mb"),
                            "memory_total_mb": gpu.get("memory_total_mb"),
                            "utilization_gpu_pct": gpu.get("utilization_gpu_pct"),
                            "power_draw_w": gpu.get("power_draw_w"),
                            "power_limit_w": gpu.get("power_limit_w"),
                        }
                        buf = self._series.setdefault(key, [])
                        buf.append(point)
                        if len(buf) > self._max_points:
                            del buf[:-self._max_points]

    def get_series(self, user: str, node: str, gpu_index: int) -> List[Dict[str, object]]:
        with self._lock:
            return list(self._series.get(self._key(user, node, gpu_index), []))


def collect_snapshot(
    users: List[str],
    partitions: Optional[List[str]],
    collector_dir: Path,
    store: CollectorStore,
) -> Dict[str, object]:
    partition_stats = get_partition_stats(partitions)
    part_index = {str(x["partition"]): x for x in partition_stats}

    running_jobs: List[Dict[str, object]] = []
    pending_jobs: List[Dict[str, object]] = []
    collectors: List[Dict[str, object]] = []

    for user in users:
        payload: Dict[str, object]
        source = "file"
        received_at = None
        cached = store.get(user)
        if cached and isinstance(cached.get("payload"), dict):
            payload = cached["payload"]  # type: ignore[assignment]
            source = "push"
            received_at = cached.get("received_at")
        else:
            path = collector_dir / f"{user}.json"
            payload = read_user_snapshot(path)

        user_running = payload.get("running_jobs", []) if isinstance(payload, dict) else []
        user_pending = payload.get("pending_jobs", []) if isinstance(payload, dict) else []

        if isinstance(user_running, list):
            running_jobs.extend(user_running)
        if isinstance(user_pending, list):
            pending_jobs.extend(user_pending)

        collectors.append(
            {
                "user": user,
                "file": str(collector_dir / f"{user}.json"),
                "source": source,
                "received_at": received_at,
                "timestamp": payload.get("timestamp") if isinstance(payload, dict) else None,
                "host": payload.get("host") if isinstance(payload, dict) else None,
                "error": payload.get("error") if isinstance(payload, dict) else "invalid collector payload",
            }
        )

    pending_by_partition: Dict[str, Dict[str, object]] = {}
    for job in pending_jobs:
        targets = job.get("partitions")
        if not isinstance(targets, list) or not targets:
            targets = [str(job.get("partition", "UNKNOWN"))]
        req = job.get("requested_gpus")
        for partition in targets:
            part_s = str(partition) if partition else "UNKNOWN"
            if part_s not in pending_by_partition:
                st = part_index.get(part_s, {})
                pending_by_partition[part_s] = {
                    "partition": part_s,
                    "pending_jobs": 0,
                    "pending_gpus": 0,
                    "node_idle_rate": st.get("node_idle_rate"),
                    "gpu_idle_rate_est": st.get("gpu_idle_rate_est"),
                    "gpu_free_est": st.get("gpu_free_est"),
                }
            pending_by_partition[part_s]["pending_jobs"] = int(pending_by_partition[part_s]["pending_jobs"]) + 1
            if isinstance(req, int):
                pending_by_partition[part_s]["pending_gpus"] = int(pending_by_partition[part_s]["pending_gpus"]) + req

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "users": users,
        "collector_dir": str(collector_dir),
        "collectors": collectors,
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "partition_stats": partition_stats,
        "pending_by_partition": sorted(pending_by_partition.values(), key=lambda x: str(x["partition"])),
    }


class SnapshotCache:
    def __init__(
        self,
        users: List[str],
        partitions: Optional[List[str]],
        refresh_seconds: int,
        collector_dir: Path,
        store: CollectorStore,
        history_store: GpuHistoryStore,
    ):
        self.users = users
        self.partitions = partitions
        self.refresh_seconds = refresh_seconds
        self.collector_dir = collector_dir
        self.store = store
        self.history_store = history_store
        self._last_ts = 0.0
        self._snapshot: Dict[str, object] = {
            "timestamp": None,
            "error": "No snapshot yet.",
        }

    def get_snapshot(self) -> Dict[str, object]:
        now = time.time()
        if now - self._last_ts >= self.refresh_seconds:
            try:
                self._snapshot = collect_snapshot(self.users, self.partitions, self.collector_dir, self.store)
                self.history_store.append_from_snapshot(self._snapshot.get("running_jobs", []), now)
            except Exception as exc:
                self._snapshot = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "error": str(exc)}
            self._last_ts = now
        return self._snapshot


class MonitorHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        cache: SnapshotCache,
        web_root: str,
        users_allowlist: List[str],
        collector_token: Optional[str],
        store: CollectorStore,
        history_store: GpuHistoryStore,
        allow_remote_ui: bool,
        **kwargs,
    ):
        self.cache = cache
        self.users_allowlist = set(users_allowlist)
        self.collector_token = collector_token
        self.store = store
        self.history_store = history_store
        self.allow_remote_ui = allow_remote_ui
        super().__init__(*args, directory=web_root, **kwargs)

    def _is_local_client(self) -> bool:
        host = self.client_address[0]
        return host in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    def do_GET(self) -> None:
        if not self.allow_remote_ui and not self._is_local_client():
            self.send_error(HTTPStatus.FORBIDDEN, "UI is local-only. Use SSH tunnel.")
            return
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
        if parsed.path == "/api/gpu-history":
            q = parse_qs(parsed.query)
            user = (q.get("user", [""])[0] or "").strip()
            node = (q.get("node", [""])[0] or "").strip()
            gpu_index_s = (q.get("gpu_index", [""])[0] or "").strip()
            if not user or not node or not gpu_index_s:
                self.send_error(HTTPStatus.BAD_REQUEST, "user,node,gpu_index are required")
                return
            gpu_index = safe_int(gpu_index_s, -1)
            if gpu_index < 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid gpu_index")
                return
            payload = {
                "user": user,
                "node": node,
                "gpu_index": gpu_index,
                "series": self.history_store.get_series(user, node, gpu_index),
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/collector":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return

        if self.collector_token:
            token = self.headers.get("X-Collector-Token", "")
            if token != self.collector_token:
                self.send_error(HTTPStatus.UNAUTHORIZED, "invalid collector token")
                return

        length_s = self.headers.get("Content-Length", "0")
        length = safe_int(length_s, 0)
        if length <= 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "empty body")
            return
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid json")
            return

        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "payload must be object")
            return
        user = str(payload.get("user", "")).strip()
        if not user:
            self.send_error(HTTPStatus.BAD_REQUEST, "missing user")
            return
        if user not in self.users_allowlist:
            self.send_error(HTTPStatus.FORBIDDEN, "user not in center --users")
            return

        self.store.update(user, payload)
        resp = json.dumps({"ok": True, "user": user}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Center web monitor for per-user collector snapshots.")
    parser.add_argument("--users", nargs="+", required=True, help="Users to aggregate.")
    parser.add_argument("--partitions", nargs="*", default=None, help="Optional partition summary scope.")
    parser.add_argument("--collector-dir", default="./collector_data", help="Directory containing <user>.json.")
    parser.add_argument("--refresh", type=int, default=10, help="API refresh interval in seconds.")
    parser.add_argument("--history-points", type=int, default=360, help="Max points kept per GPU history.")
    parser.add_argument("--collector-token", default=None, help="Optional shared token for collector POST.")
    parser.add_argument("--allow-remote-ui", action="store_true", help="Allow remote clients to access UI and GET APIs.")
    parser.add_argument("--serve", action="store_true", help="Start web server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=18080, help="Bind port.")
    return parser.parse_args()


def main() -> int:
    for cmd in ["squeue", "sinfo"]:
        require_cmd(cmd)
    args = parse_args()
    collector_dir = Path(args.collector_dir).resolve()
    store = CollectorStore()
    history_store = GpuHistoryStore(max(args.history_points, 60))
    if not args.serve:
        print(json.dumps(collect_snapshot(args.users, args.partitions, collector_dir, store), indent=2, ensure_ascii=False))
        return 0

    web_root = str(Path(__file__).resolve().parent / "web")
    cache = SnapshotCache(
        args.users,
        args.partitions,
        max(args.refresh, 3),
        collector_dir,
        store,
        history_store,
    )

    def handler(*handler_args, **handler_kwargs):
        return MonitorHandler(
            *handler_args,
            cache=cache,
            web_root=web_root,
            users_allowlist=args.users,
            collector_token=args.collector_token,
            store=store,
            history_store=history_store,
            allow_remote_ui=args.allow_remote_ui,
            **handler_kwargs,
        )

    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving dashboard at http://{args.host}:{args.port}")
    print(f"Collector dir: {collector_dir}")
    print("Collector push endpoint: POST /api/collector")
    if args.allow_remote_ui:
        print("UI mode: remote access enabled")
    else:
        print("UI mode: local-only (use SSH tunnel)")
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
