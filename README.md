# Slurm Multi-User GPU Monitor

This directory implements a multi-user Slurm monitoring system with:

- per-user collectors (run under each user account),
- one center web service (aggregates and visualizes data),
- a web dashboard for running jobs, GPU metrics, pending jobs, and partition idle rate.

## Files

- `user_gpu_collector.py`: collector process (run by each user)
- `start_user_collector.sh`: launcher for collector
- `slurm_monitor_center.py`: center service and web API
- `start_monitor_web.sh`: launcher for center service
- `web/index.html`: dashboard UI

## Architecture

1. Each user runs a collector in `tmux`.
2. Collector gathers that user's:
   - running jobs (`squeue`)
   - pending jobs (`squeue`)
   - node GPU stats (`nvidia-smi` via ssh to allocated nodes)
3. Collector pushes JSON to center (`POST /api/collector`).
4. Center aggregates all users and serves dashboard/API.

## Start Collector (Per User)

Run under each user account:

```bash
cd Node_Monitor
tmux new -s mon_<user>
./start_user_collector.sh <user> \
  --interval 10 \
  --gpu-timeout 30 \
  --ssh-connect-timeout 8 \
  --gpu-retries 2 \
  --gpu-retry-delay 1.5 \
  --push-url http://<center-host>:18080/api/collector \
  --push-token <token>
```

Example:

```bash
./start_user_collector.sh user_a \
  --interval 10 \
  --gpu-timeout 30 \
  --ssh-connect-timeout 8 \
  --gpu-retries 2 \
  --gpu-retry-delay 1.5 \
  --push-url http://172.11.11.1:18080/api/collector \
  --push-token mytoken
```

## Start Center

Run on center host:

```bash
cd Node_Monitor
./start_monitor_web.sh user_a user_b \
  --port 18080 \
  --refresh 10 \
  --collector-token mytoken \
  --history-points 360
```

## Access Control (SSH-Only UI)

Current default behavior:

- UI and `GET` APIs are local-only on center (remote clients get 403).
- Collector `POST /api/collector` is still allowed remotely.

Recommended access:

```bash
ssh -L 18080:127.0.0.1:18080 <user>@<center-host>
```

Then open locally:

- `http://127.0.0.1:18080`

If you need remote UI access (not recommended by default), add:

```bash
--allow-remote-ui
```

## Dashboard Sections

- **Collector Status**
  - user, host, collector timestamp, center receive time, source, error state
- **Running Jobs & GPU Metrics**
  - grouped by user, expandable job blocks, nvitop-style bars (MEM / UTIL / PWR)
  - per-GPU history chart (fixed y-scale by metric type)
- **Partition Idle Rate**
  - grouped by user
- **Pending Jobs**
  - global table

## Collector Timeout / Retry Options

Collector supports the following options for unstable nodes:

- `--gpu-timeout`: timeout for one GPU query attempt (seconds)
- `--ssh-connect-timeout`: SSH connection timeout (seconds)
- `--gpu-retries`: retry count when GPU query fails
- `--gpu-retry-delay`: delay between retries (seconds)
- `--no-stale-on-error`: disable stale fallback

Default behavior includes stale fallback:

- If a node query fails after retries, collector can reuse the last successful
  GPU metrics for that node and mark them as stale.
- This keeps dashboard stable during transient SSH/NVIDIA hiccups.

## Troubleshooting

- `unrecognized arguments: --gpu-timeout ...`
  - collector script on that account is old version; verify path and run
    `python3 user_gpu_collector.py --help`.
- `invalid collector token`
  - ensure collector `--push-token` matches center `--collector-token`.
- no GPU metrics / timeout on `ssh ... nvidia-smi`
  - check SSH from that user to allocated nodes and `nvidia-smi` availability.
  - increase `--gpu-timeout`, `--ssh-connect-timeout`, and `--gpu-retries`.
- empty dashboard
  - verify collectors are running and pushing to the correct center URL.

