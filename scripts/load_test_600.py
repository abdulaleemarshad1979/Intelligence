#!/usr/bin/env python3
"""Progressive Load Testing & Capacity Benchmark Harness for 600 CCTV Camera Fleet.

Executes progressive scale tests across:
1 -> 5 -> 10 -> 25 -> 50 -> 100 -> 200 -> 300 -> 400 -> 500 -> 600 cameras.

Uses Python standard library and Linux /proc metrics (zero extra dependencies).
"""

import os
import sys
import time
import threading
from typing import Dict, Any, List, Tuple

# Ensure app imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.stream_manager import get_stream_manager, ai_worker_pool
from app.ingestion.mediamtx_manager import mediamtx_mgr


def get_process_metrics() -> Tuple[float, float, int]:
    """Read RSS memory (MB), CPU time, and thread count from Linux /proc filesystem."""
    rss_mb = 0.0
    threads = threading.active_count()

    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = float(line.split()[1])
                    rss_mb = rss_kb / 1024.0
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
    except Exception:
        pass

    return rss_mb, threads


def get_cpu_times() -> Tuple[float, float]:
    """Read process CPU user/system times."""
    t = os.times()
    return t.user + t.system, time.time()


def run_progressive_benchmark(target_levels: List[int] = None):
    if target_levels is None:
        target_levels = [1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 600]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    stream_mgr = get_stream_manager(data_dir)

    results = []
    num_cpus = os.cpu_count() or 16

    print("=" * 82)
    print("  ANDHRA PRADESH POLICE CCTV PLATFORM - 600-CAMERA ARCHITECTURE BENCHMARK")
    print(f"  CPU Logical Cores: {num_cpus}")
    print(f"  MediaMTX Streaming Layer: {'ONLINE' if mediamtx_mgr.is_api_alive() else 'STANDBY'}")
    print("=" * 82)
    print(f"{'Cameras':<9} | {'App CPU%':<10} | {'App RAM':<10} | {'Threads':<8} | {'MediaMTX':<10} | {'AI Queue':<9} | {'Latency':<8}")
    print("-" * 82)

    for count in target_levels:
        # Attach / ensure workers exist up to target count
        for i in range(1, count + 1):
            cid = f"CAM-{i:03d}"
            if cid not in stream_mgr.workers:
                stream_mgr.get_or_create_worker(cid)

        # Measure CPU time over interval
        cpu_t0, wall_t0 = get_cpu_times()
        time.sleep(0.6)
        cpu_t1, wall_t1 = get_cpu_times()

        wall_dt = max(0.001, wall_t1 - wall_t0)
        cpu_dt = max(0.0, cpu_t1 - cpu_t0)
        app_cpu = (cpu_dt / wall_dt) * 100.0

        app_ram_mb, thread_count = get_process_metrics()
        mtx_stats = mediamtx_mgr.get_stats()
        ai_q_size = ai_worker_pool.task_queue.qsize()

        # Measure snapshot pull latency
        t0 = time.perf_counter()
        w = stream_mgr.workers.get("CAM-001")
        if w:
            _ = w.get_jpeg_frame(quality=80)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        mtx_tag = "ONLINE" if mtx_stats["is_mediamtx_live"] else "STANDBY"

        print(f"{count:<9} | {app_cpu:<9.1f}% | {app_ram_mb:<7.1f} MB | {thread_count:<8} | {mtx_tag:<10} | {ai_q_size:<9} | {latency_ms:<6.1f}ms")

        results.append({
            "cameras": count,
            "app_cpu_pct": app_cpu,
            "app_ram_mb": round(app_ram_mb, 1),
            "threads": thread_count,
            "mediamtx_online": mtx_stats["is_mediamtx_live"],
            "ai_queue_size": ai_q_size,
            "latency_ms": round(latency_ms, 2)
        })

    print("=" * 82)
    print("  600-CAMERA ARCHITECTURE BENCHMARK COMPLETE")
    print(f"  Final Active Camera Workers: {len(stream_mgr.workers)}")
    print(f"  AI Worker Pool Status: {ai_worker_pool.max_workers} Bounded Workers | Stale Drops: {ai_worker_pool.total_tasks_dropped}")
    print("=" * 82)
    return results


if __name__ == "__main__":
    run_progressive_benchmark()
