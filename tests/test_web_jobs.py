from __future__ import annotations

import sys
import time

from skull.web.jobs import (
    build_pytest_command,
    build_script_command,
    build_train_command,
    delete_job,
    find_job,
    log_path,
    read_log_tail,
    request_stop,
    start_job,
)


def _wait_for_status(root, job_id: str, statuses: set[str], timeout: float = 20.0):
    deadline = time.time() + timeout
    latest = None
    while time.time() < deadline:
        latest = find_job(root, job_id)
        if latest and latest.get("status") in statuses:
            return latest
        time.sleep(0.25)
    raise AssertionError(
        f"Timed out waiting for job {job_id} to reach {statuses}. Last state: {latest}"
    )


def test_web_job_completes_and_writes_log(tmp_path):
    job = start_job(
        tmp_path,
        job_type="test",
        label="echo",
        command=[sys.executable, "-c", "print('hello from web job')"],
        metadata={"kind": "unit"},
    )

    finished = _wait_for_status(tmp_path, job["id"], {"completed"})

    assert finished["returncode"] == 0
    assert finished["finished_at"] is not None
    assert finished["metadata"]["kind"] == "unit"
    assert "hello from web job" in read_log_tail(finished)


def test_web_job_can_be_stopped(tmp_path):
    job = start_job(
        tmp_path,
        job_type="test",
        label="sleepy",
        command=[
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(20)",
        ],
    )

    _wait_for_status(tmp_path, job["id"], {"running", "starting"})
    request_stop(tmp_path, job["id"])
    stopped = _wait_for_status(tmp_path, job["id"], {"stopped"})

    assert stopped["finished_at"] is not None
    assert "stop requested" in read_log_tail(stopped).lower()


def test_web_job_stop_allows_graceful_exit(tmp_path):
    job = start_job(
        tmp_path,
        job_type="train:pretrain",
        label="graceful",
        command=[
            sys.executable,
            "-c",
            (
                "import os, pathlib, time; "
                "stop_path = pathlib.Path(os.environ['SKULL_STOP_REQUEST_PATH']); "
                "print('started', flush=True); "
                "while not stop_path.exists(): time.sleep(0.1); "
                "print('noticed stop', flush=True)"
            ),
        ],
    )

    _wait_for_status(tmp_path, job["id"], {"running", "starting"})
    request_stop(tmp_path, job["id"])
    stopped = _wait_for_status(tmp_path, job["id"], {"stopped"})

    log_tail = read_log_tail(stopped).lower()
    assert stopped["finished_at"] is not None
    assert "noticed stop" in log_tail
    assert "force killing" not in log_tail


def test_web_job_can_be_deleted_after_it_finishes(tmp_path):
    job = start_job(
        tmp_path,
        job_type="test",
        label="delete-me",
        command=[sys.executable, "-c", "print('done')"],
    )
    finished = _wait_for_status(tmp_path, job["id"], {"completed"})

    removed = delete_job(tmp_path, finished["id"])

    assert removed["id"] == finished["id"]
    assert find_job(tmp_path, finished["id"]) is None
    assert log_path(tmp_path, finished["id"]).exists()


def test_web_job_delete_can_remove_log(tmp_path):
    job = start_job(
        tmp_path,
        job_type="test",
        label="delete-log",
        command=[sys.executable, "-c", "print('done')"],
    )
    finished = _wait_for_status(tmp_path, job["id"], {"completed"})

    delete_job(tmp_path, finished["id"], delete_log=True)

    assert find_job(tmp_path, finished["id"]) is None
    assert not log_path(tmp_path, finished["id"]).exists()


def test_web_job_delete_rejects_active_job(tmp_path):
    job = start_job(
        tmp_path,
        job_type="test",
        label="active",
        command=[sys.executable, "-c", "import time; time.sleep(20)"],
    )
    running = _wait_for_status(tmp_path, job["id"], {"running", "starting"})

    try:
        delete_job(tmp_path, running["id"])
    except RuntimeError as exc:
        assert "stopped" in str(exc)
    else:
        raise AssertionError("delete_job should reject active jobs")
    finally:
        request_stop(tmp_path, running["id"])
        _wait_for_status(tmp_path, running["id"], {"stopped"})


def test_build_train_command_supports_accelerate():
    command = build_train_command(
        "pretrain",
        "configs/train/pretrain_150m.yaml",
        use_accelerate=True,
        num_processes=2,
    )

    assert command[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    assert "--num_processes" in command
    assert "--accelerate" in command


def test_build_script_command_uses_current_python():
    command = build_script_command(
        "scripts/count_tokens.py",
        ["--input", "data/clean/demo.txt"],
    )

    assert command == [
        sys.executable,
        "scripts/count_tokens.py",
        "--input",
        "data/clean/demo.txt",
    ]


def test_build_pytest_command_accepts_targets_and_extra_args():
    command = build_pytest_command(["tests/test_demo.py"], ["-q"])

    assert command == [sys.executable, "-m", "pytest", "tests/test_demo.py", "-q"]
