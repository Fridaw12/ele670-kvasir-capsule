#!/usr/bin/env python3
"""
Pick a GPU that is (a) not used by other users and (b) has the most free memory.
Prints the chosen GPU *index* to stdout. Exits non-zero if none are safe.

Usage:
  ./pick_gpu_safe.py
  CUDA_VISIBLE_DEVICES=$(./pick_gpu_safe.py) python train.py
"""

import subprocess
import sys
import os
import getpass

def run(cmd):
    return subprocess.check_output(cmd, encoding="utf-8").strip()

def parse_csv_lines(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [ [c.strip() for c in line.split(",")] for line in lines ]

def main():
    me = getpass.getuser()

    # Map GPU index -> UUID and free/used memory
    gpu_info = parse_csv_lines(run([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits"
    ]))
    # Example row: ['0', 'GPU-xxxx', '16160', '0', '16160']
    idx_uuid = {int(r[0]): r[1] for r in gpu_info}
    mem_free = {int(r[0]): int(r[4]) for r in gpu_info}
    gpu_indices = sorted(idx_uuid.keys())

    # Query all compute processes on GPUs (PID + which GPU UUID)
    # (graphics processes are usually irrelevant for headless servers)
    try:
        proc_rows = parse_csv_lines(run([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits"
        ]))
    except subprocess.CalledProcessError:
        proc_rows = []  # no processes

    # Build: GPU index -> set of usernames who have processes there
    users_on_gpu = {i: set() for i in gpu_indices}
    for row in proc_rows:
        if len(row) < 2:
            continue
        gpu_uuid, pid_s = row[0], row[1]
        # Map UUID back to index
        gpu_idx = next((i for i,u in idx_uuid.items() if u == gpu_uuid), None)
        if gpu_idx is None:
            continue
        # Find the username for this PID
        try:
            u = run(["ps", "-o", "user=", "-p", pid_s]).split()[0]
        except Exception:
            continue
        users_on_gpu[gpu_idx].add(u)

    # Keep GPUs that have either:
    #  - no processes at all, or
    #  - only *your* processes
    safe_gpus = []
    for i in gpu_indices:
        others = {u for u in users_on_gpu[i] if u != me}
        if not others:
            safe_gpus.append(i)

    if not safe_gpus:
        print("No safe GPU available (others are in use).", file=sys.stderr)
        sys.exit(2)

    # Among safe GPUs, pick the one with the most free memory
    best = max(safe_gpus, key=lambda i: mem_free.get(i, -1))
    print(best)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("Error: nvidia-smi not found or NVIDIA drivers not available.", file=sys.stderr)
        sys.exit(3)
    except subprocess.CalledProcessError as e:
        print(f"Error while querying GPUs: {e}", file=sys.stderr)
        sys.exit(4)
