from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ACTIVE_JOB_STATUSES = {"starting", "running", "stop_requested", "stopping"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def state_root(root: str | Path) -> Path:
    path = Path(root).resolve() / ".skull_web"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_root(root: str | Path) -> Path:
    path = state_root(root) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def requests_root(root: str | Path) -> Path:
    path = state_root(root) / "requests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def registry_path(root: str | Path) -> Path:
    return state_root(root) / "jobs.json"


def log_path(root: str | Path, job_id: str) -> Path:
    return logs_root(root) / f"{job_id}.log"


def stop_request_path(root: str | Path, job_id: str) -> Path:
    return requests_root(root) / f"{job_id}.stop"


class _RegistryLock:
    def __init__(self, root: str | Path) -> None:
        self.path = state_root(root) / ".jobs.lock"

    def __enter__(self) -> "_RegistryLock":
        deadline = time.time() + 5.0
        while True:
            try:
                fd = os.open(
                    str(self.path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    try:
                        if self.path.exists() and (
                            time.time() - self.path.stat().st_mtime
                        ) > 30.0:
                            self.path.unlink()
                            continue
                    except FileNotFoundError:
                        continue
                    raise TimeoutError("Timed out waiting for job registry lock.")
                time.sleep(0.05)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _load_registry_unlocked(root: str | Path) -> list[dict[str, Any]]:
    path = registry_path(root)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    jobs = payload.get("jobs", [])
    return jobs if isinstance(jobs, list) else []


def _write_registry_unlocked(root: str | Path, jobs: list[dict[str, Any]]) -> None:
    path = registry_path(root)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"jobs": jobs}, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def _upsert_job_unlocked(root: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    jobs = _load_registry_unlocked(root)
    updated = False
    for index, job in enumerate(jobs):
        if job.get("id") == record["id"]:
            jobs[index] = {**job, **record}
            updated = True
            break
    if not updated:
        jobs.append(record)
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    _write_registry_unlocked(root, jobs)
    return next(job for job in jobs if job.get("id") == record["id"])


def load_jobs(root: str | Path) -> list[dict[str, Any]]:
    with _RegistryLock(root):
        jobs = _load_registry_unlocked(root)

    refreshed: list[dict[str, Any]] = []
    for job in jobs:
        status = str(job.get("status", "unknown"))
        runner_pid = job.get("runner_pid")
        if (
            status in ACTIVE_JOB_STATUSES
            and isinstance(runner_pid, int)
            and not is_process_running(runner_pid)
        ):
            job = {
                **job,
                "status": "unknown",
                "finished_at": job.get("finished_at") or _now(),
            }
        refreshed.append(job)

    refreshed.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return refreshed


def save_job(root: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    with _RegistryLock(root):
        return _upsert_job_unlocked(root, record)


def update_job(root: str | Path, job_id: str, **fields: Any) -> dict[str, Any]:
    with _RegistryLock(root):
        jobs = _load_registry_unlocked(root)
        current = next((job for job in jobs if job.get("id") == job_id), None)
        if current is None:
            raise KeyError(f"Unknown job id: {job_id}")
        updated = {**current, **fields}
        return _upsert_job_unlocked(root, updated)


def find_job(root: str | Path, job_id: str) -> dict[str, Any] | None:
    for job in load_jobs(root):
        if job.get("id") == job_id:
            return job
    return None


def _unlink_with_retries(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.time() >= deadline:
                raise
            time.sleep(0.1)


def delete_job(
    root: str | Path,
    job_id: str,
    *,
    delete_log: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    job = find_job(repo_root, job_id)
    if job is None:
        raise KeyError(f"Unknown job id: {job_id}")
    if job.get("status") in ACTIVE_JOB_STATUSES:
        raise RuntimeError("Active jobs must be stopped before deletion.")

    with _RegistryLock(repo_root):
        jobs = _load_registry_unlocked(repo_root)
        remaining = [item for item in jobs if item.get("id") != job_id]
        if len(remaining) == len(jobs):
            raise KeyError(f"Unknown job id: {job_id}")
        _write_registry_unlocked(repo_root, remaining)

    clear_stop_request(repo_root, job_id)
    if delete_log:
        _unlink_with_retries(log_path(repo_root, job_id))
    return job


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def is_process_running(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False

    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = result.stdout.strip().lower()
        return (
            str(pid) in stdout
            and "no tasks are running" not in stdout
            and "info:" not in stdout
        )
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_train_command(
    mode: str,
    config_path: str,
    *,
    use_accelerate: bool = False,
    num_processes: int | None = None,
) -> list[str]:
    if mode not in {"pretrain", "cpt", "sft"}:
        raise ValueError(f"Unsupported train mode: {mode}")
    if use_accelerate:
        command = [sys.executable, "-m", "accelerate.commands.launch"]
        if num_processes is not None:
            command.extend(["--num_processes", str(int(num_processes))])
        command.extend(
            ["-m", f"skull.cli.{mode}", "--config", config_path, "--accelerate"]
        )
        return command
    return [sys.executable, "-m", f"skull.cli.{mode}", "--config", config_path]


def build_eval_command(
    config_path: str,
    checkpoint_path: str,
    *,
    print_json: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "skull.cli.eval",
        "--config",
        config_path,
        "--ckpt",
        checkpoint_path,
    ]
    if print_json:
        command.append("--print_json")
    return command


def build_sample_command(
    config_path: str,
    checkpoint_path: str,
    *,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "skull.cli.sample",
        "--config",
        config_path,
        "--ckpt",
        checkpoint_path,
        "--prompt",
        prompt,
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--temperature",
        str(float(temperature)),
    ]
    if top_k is not None:
        command.extend(["--top_k", str(int(top_k))])
    return command


def build_script_command(script_path: str, args: list[str] | None = None) -> list[str]:
    command = [sys.executable, script_path]
    command.extend(str(arg) for arg in (args or []) if str(arg) != "")
    return command


def build_pytest_command(
    targets: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [sys.executable, "-m", "pytest"]
    command.extend(str(target) for target in (targets or []) if str(target) != "")
    command.extend(str(arg) for arg in (extra_args or []) if str(arg) != "")
    return command


def start_job(
    root: str | Path,
    *,
    job_type: str,
    label: str,
    command: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    job_id = uuid.uuid4().hex[:12]
    created_at = _now()
    record = {
        "id": job_id,
        "job_type": job_type,
        "label": label,
        "status": "starting",
        "created_at": created_at,
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "command": command,
        "display_command": format_command(command),
        "runner_pid": None,
        "child_pid": None,
        "log_path": str(log_path(repo_root, job_id).resolve()),
        "repo_root": str(repo_root),
        "metadata": metadata or {},
    }
    save_job(repo_root, record)

    runner_command = [
        sys.executable,
        "-m",
        "skull.web.job_runner",
        "--root",
        str(repo_root),
        "--job-id",
        job_id,
        "--",
        *command,
    ]

    job_log_path = log_path(repo_root, job_id)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    start_new_session = os.name != "nt"

    with open(job_log_path, "a", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            runner_command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )

    updated = update_job(
        repo_root,
        job_id,
        runner_pid=process.pid,
        runner_command=runner_command,
    )
    return updated


def request_stop(root: str | Path, job_id: str) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    job = find_job(repo_root, job_id)
    if job is None:
        raise KeyError(f"Unknown job id: {job_id}")

    stop_path = stop_request_path(repo_root, job_id)
    stop_path.write_text("stop\n", encoding="utf-8")
    return update_job(
        repo_root,
        job_id,
        status="stop_requested",
        stop_requested_at=_now(),
    )


def clear_stop_request(root: str | Path, job_id: str) -> None:
    try:
        stop_request_path(root, job_id).unlink()
    except FileNotFoundError:
        pass


def read_log_tail(job: dict[str, Any], *, max_chars: int = 20000) -> str:
    path = Path(str(job.get("log_path", "")))
    if not path.exists():
        return ""

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        chunk_size = min(size, max_chars * 4)
        f.seek(max(0, size - chunk_size))
        text = f.read().decode("utf-8", errors="replace")

    if len(text) <= max_chars:
        return text
    return "...\n" + text[-max_chars:]


def current_time_iso() -> str:
    return _now()
