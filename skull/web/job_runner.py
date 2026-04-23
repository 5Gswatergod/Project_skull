from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from skull.web.jobs import (
    clear_stop_request,
    current_time_iso,
    find_job,
    format_command,
    stop_request_path,
    update_job,
)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


GRACEFUL_STOP_TIMEOUT_SEC = _float_env("SKULL_GRACEFUL_STOP_TIMEOUT_SEC", 180.0)
TERMINATE_TIMEOUT_SEC = _float_env("SKULL_TERMINATE_TIMEOUT_SEC", 6.0)


def _terminate_process(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    os.killpg(pid, signal.SIGTERM)


def _force_kill_process(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    os.killpg(pid, signal.SIGKILL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Skull web job runner")
    parser.add_argument("--root", required=True, help="Repository root path")
    parser.add_argument("--job-id", required=True, help="Registry job id")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No target command provided to job runner.")

    repo_root = Path(args.root).resolve()
    job_id = args.job_id
    job_record = find_job(repo_root, job_id) or {}
    graceful_stop_supported = str(job_record.get("job_type", "")).startswith("train:")

    update_job(
        repo_root,
        job_id,
        status="running",
        started_at=current_time_iso(),
        runner_pid=os.getpid(),
    )

    print(f"[web-job] starting job={job_id}", flush=True)
    print(f"[web-job] repo={repo_root}", flush=True)
    print(f"[web-job] command={format_command(command)}", flush=True)

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    start_new_session = os.name != "nt"
    stop_path = stop_request_path(repo_root, job_id)

    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "SKULL_STOP_REQUEST_PATH": str(stop_path),
        },
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )

    update_job(repo_root, job_id, child_pid=process.pid)

    stop_requested = False
    stop_deadline: float | None = None
    terminate_sent = False
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                break

            if stop_path.exists() and not stop_requested:
                stop_requested = True
                update_job(repo_root, job_id, status="stopping")
                if graceful_stop_supported:
                    stop_deadline = time.time() + GRACEFUL_STOP_TIMEOUT_SEC
                    print(
                        "[web-job] stop requested for job="
                        f"{job_id}; waiting for graceful shutdown",
                        flush=True,
                    )
                else:
                    terminate_sent = True
                    stop_deadline = time.time() + TERMINATE_TIMEOUT_SEC
                    print(
                        "[web-job] stop requested for job="
                        f"{job_id}; terminating process tree",
                        flush=True,
                    )
                    _terminate_process(process.pid)

            if (
                stop_requested
                and stop_deadline is not None
                and time.time() >= stop_deadline
            ):
                if graceful_stop_supported and not terminate_sent:
                    terminate_sent = True
                    stop_deadline = time.time() + TERMINATE_TIMEOUT_SEC
                    print(
                        "[web-job] graceful shutdown timed out for job="
                        f"{job_id}; terminating process tree",
                        flush=True,
                    )
                    _terminate_process(process.pid)
                else:
                    print(
                        "[web-job] terminate timed out for job="
                        f"{job_id}; force killing process tree",
                        flush=True,
                    )
                    _force_kill_process(process.pid)
                    break

            time.sleep(0.5)

        returncode = process.wait()
        clear_stop_request(repo_root, job_id)

        if stop_requested:
            status = "stopped"
        elif returncode == 0:
            status = "completed"
        else:
            status = "failed"

        update_job(
            repo_root,
            job_id,
            status=status,
            finished_at=current_time_iso(),
            returncode=returncode,
        )
        print(
            f"[web-job] finished job={job_id} status={status} returncode={returncode}",
            flush=True,
        )
        return int(returncode)
    except BaseException:
        clear_stop_request(repo_root, job_id)
        update_job(
            repo_root,
            job_id,
            status="failed",
            finished_at=current_time_iso(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
