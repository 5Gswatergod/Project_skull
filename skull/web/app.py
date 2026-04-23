from __future__ import annotations

import html
import os
import shlex
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from skull.web.data import collect_dashboard_state
from skull.web.jobs import (
    ACTIVE_JOB_STATUSES,
    build_eval_command,
    build_pytest_command,
    build_sample_command,
    build_train_command,
    delete_job,
    load_jobs,
    read_log_tail,
    request_stop,
    start_job,
)


DEFAULT_REPO_ROOT = Path(
    os.environ.get("SKULL_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()

PAGES = ["Home", "Run", "Monitor", "Assets", "Help"]
PAGE_STATE_KEY = "page"
PENDING_PAGE_STATE_KEY = "_pending_page"


@st.cache_data(show_spinner=False, ttl=2)
def _load_state(repo_root: str) -> dict[str, Any]:
    return collect_dashboard_state(repo_root)


def _safe_index(options: list[str], preferred: str | None = None) -> int:
    if not options:
        return 0
    if preferred and preferred in options:
        return options.index(preferred)
    return 0


def _repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _split_cli_args(value: str) -> list[str]:
    if not value.strip():
        return []
    return shlex.split(value, posix=os.name != "nt")


def _nonempty_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _path_exists(repo_root: Path, path_value: str, label: str) -> bool:
    path = _repo_path(repo_root, path_value)
    if not path_value or not path.exists():
        st.error(f"{label} not found: {path}")
        return False
    return True


def _normalize_run_dir(path_value: str | None) -> str | None:
    if not path_value:
        return None
    return Path(str(path_value)).as_posix().lstrip("./")


def _train_configs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in state["configs"] if item.get("kind") == "train"]


def _eval_configs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in state["configs"] if item.get("kind") == "eval"]


def _checkpoint_paths(state: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for run in state["runs"]:
        for checkpoint in run.get("checkpoints", []):
            path = checkpoint.get("relative_path")
            if path:
                paths.append(str(path))
    return sorted(dict.fromkeys(paths))


def _train_config_record(
    state: dict[str, Any],
    path_value: str,
) -> dict[str, Any] | None:
    return next(
        (
            config
            for config in _train_configs(state)
            if config.get("relative_path") == path_value
        ),
        None,
    )


def _infer_train_mode(
    config: dict[str, Any] | None,
    config_path: str,
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    content = config.get("content", {}) if config else {}
    path_text = config_path.replace("\\", "/").lower()
    run_dir = str(content.get("run_dir", "")).replace("\\", "/").lower()

    if "train_jsonl" in content or "/sft/" in run_dir or "sft" in path_text:
        return "sft"
    if "base_ckpt" in content or "/cpt/" in run_dir or "cpt" in path_text:
        return "cpt"
    return "pretrain"


def _referenced_train_paths(
    config: dict[str, Any] | None,
) -> list[tuple[str, str, bool]]:
    if not config:
        return []

    content = config.get("content", {})
    references: list[tuple[str, str, bool]] = []

    for key in ["tokenizer_model", "model_config", "base_ckpt"]:
        value = content.get(key)
        if value:
            references.append((key, str(value), True))

    for key in ["train_jsonl", "val_jsonl"]:
        value = content.get(key)
        if value:
            references.append((key, str(value), True))

    for section in ["train_sources", "val_sources"]:
        for source in content.get(section) or []:
            if not isinstance(source, dict):
                continue
            source_name = source.get("name", section)
            for path in source.get("paths") or []:
                references.append((f"{section}:{source_name}", str(path), True))

    return references


def _missing_train_config_paths(
    repo_root: Path,
    config: dict[str, Any] | None,
) -> list[dict[str, str]]:
    missing = []
    for label, path_value, required in _referenced_train_paths(config):
        path = _repo_path(repo_root, path_value)
        if required and not path.exists():
            missing.append(
                {
                    "field": label,
                    "path": path_value,
                    "resolved_path": str(path),
                }
            )
    return missing


def _match_run_config(
    state: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any] | None:
    run_path = run.get("relative_path")
    run_name = run.get("name")

    for config in _train_configs(state):
        if _normalize_run_dir(config.get("run_dir")) == run_path:
            return config

    for config in _train_configs(state):
        if config.get("run_name") == run_name:
            return config

    return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "n/a"

    seconds = int(max(0, float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _active_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [job for job in jobs if job.get("status") in ACTIVE_JOB_STATUSES]


def _set_page(page: str) -> None:
    if page in PAGES:
        st.session_state[PENDING_PAGE_STATE_KEY] = page


def _sync_pending_page() -> None:
    if PAGE_STATE_KEY not in st.session_state:
        st.session_state[PAGE_STATE_KEY] = "Home"

    pending_page = st.session_state.pop(PENDING_PAGE_STATE_KEY, None)
    if pending_page in PAGES:
        st.session_state[PAGE_STATE_KEY] = pending_page


def _set_notice(kind: str, message: str) -> None:
    st.session_state["notice"] = {"kind": kind, "message": message}


def _display_notice() -> None:
    notice = st.session_state.pop("notice", None)
    if not notice:
        return
    kind = str(notice.get("kind", "info"))
    message = str(notice.get("message", ""))
    renderer = getattr(st, kind, st.info)
    renderer(message)
    st.toast(message)


def _start_and_focus(
    repo_root: Path,
    *,
    job_type: str,
    label: str,
    command: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    job = start_job(
        repo_root,
        job_type=job_type,
        label=label,
        command=command,
        metadata=metadata,
    )
    _set_page("Monitor")
    _set_notice("success", f"Started {job['label']} ({job['id']}).")
    st.cache_data.clear()
    st.rerun()


def _badge(text: str, tone: str) -> str:
    safe_tone = tone if tone in {"good", "warn", "bad", "info", "neutral"} else "neutral"
    return f"<span class='sk-badge sk-badge-{safe_tone}'>{html.escape(text)}</span>"


def _status_tone(status: str | None) -> str:
    if status in {"completed"}:
        return "good"
    if status in ACTIVE_JOB_STATUSES:
        return "info"
    if status in {"failed", "unknown"}:
        return "bad"
    if status in {"stopped"}:
        return "warn"
    return "neutral"


def _table(data: list[dict[str, Any]], *, height: int = 300) -> None:
    if not data:
        st.info("Nothing to show.")
        return
    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        height=height,
        hide_index=True,
    )


def _theme_tokens(appearance: str) -> str:
    light_tokens = """
            --sk-text: #111827;
            --sk-muted: #475569;
            --sk-border: #cbd5e1;
            --sk-panel: #ffffff;
            --sk-panel-strong: #f8fafc;
            --sk-bg: #f3f6fa;
            --sk-sidebar: #ffffff;
            --sk-input: #ffffff;
            --sk-accent: #dc2626;
            --sk-accent-hover: #b91c1c;
            --sk-focus: #2563eb;
            --sk-shadow: rgba(15, 23, 42, 0.08);
            --sk-good-text: #065f46;
            --sk-good-bg: #d1fae5;
            --sk-good-border: #a7f3d0;
            --sk-warn-text: #92400e;
            --sk-warn-bg: #fef3c7;
            --sk-warn-border: #fde68a;
            --sk-bad-text: #991b1b;
            --sk-bad-bg: #fee2e2;
            --sk-bad-border: #fecaca;
            --sk-info-text: #1e40af;
            --sk-info-bg: #dbeafe;
            --sk-info-border: #bfdbfe;
            --sk-neutral-text: #334155;
            --sk-neutral-bg: #f1f5f9;
            --sk-neutral-border: #cbd5e1;
    """
    dark_tokens = """
            --sk-text: #f8fafc;
            --sk-muted: #cbd5e1;
            --sk-border: #334155;
            --sk-panel: #111827;
            --sk-panel-strong: #0f172a;
            --sk-bg: #020617;
            --sk-sidebar: #0b1120;
            --sk-input: #0f172a;
            --sk-accent: #f97373;
            --sk-accent-hover: #fb7185;
            --sk-focus: #60a5fa;
            --sk-shadow: rgba(0, 0, 0, 0.35);
            --sk-good-text: #bbf7d0;
            --sk-good-bg: #064e3b;
            --sk-good-border: #047857;
            --sk-warn-text: #fde68a;
            --sk-warn-bg: #713f12;
            --sk-warn-border: #a16207;
            --sk-bad-text: #fecaca;
            --sk-bad-bg: #7f1d1d;
            --sk-bad-border: #b91c1c;
            --sk-info-text: #bfdbfe;
            --sk-info-bg: #1e3a8a;
            --sk-info-border: #2563eb;
            --sk-neutral-text: #e2e8f0;
            --sk-neutral-bg: #1e293b;
            --sk-neutral-border: #475569;
    """

    if appearance == "Dark":
        return dark_tokens
    if appearance == "Light":
        return light_tokens
    return (
        light_tokens
        + """
        }
        @media (prefers-color-scheme: dark) {
        :root {
        """
        + dark_tokens
    )


def _inject_styles(appearance: str) -> None:
    close_auto_media = "\n        }\n        }" if appearance == "Auto" else "\n        }"
    css = """
        <style>
        :root {
        __SK_TOKENS__
        __SK_CLOSE__
        .stApp {
            background: var(--sk-bg);
            color: var(--sk-text);
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        .main .block-container {
            background: var(--sk-bg);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: var(--sk-text);
            letter-spacing: 0;
        }
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp small,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMarkdownContainer"] {
            color: var(--sk-text);
        }
        .stCaptionContainer,
        [data-testid="stCaptionContainer"],
        [data-testid="stMetricLabel"] {
            color: var(--sk-muted) !important;
        }
        p, li, label, span {
            letter-spacing: 0;
        }
        section[data-testid="stSidebar"] {
            background: var(--sk-sidebar);
            border-right: 1px solid var(--sk-border);
        }
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--sk-text) !important;
        }
        div[data-testid="stMetric"] {
            background: var(--sk-panel);
            border: 1px solid var(--sk-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 28px var(--sk-shadow);
        }
        .stButton > button,
        .stFormSubmitButton > button {
            min-height: 48px;
            border-radius: 8px;
            font-weight: 700;
        }
        .stButton button[data-testid="baseButton-secondary"],
        .stFormSubmitButton button[data-testid="baseButton-secondary"],
        .stButton button[kind="secondary"],
        .stFormSubmitButton button[kind="secondary"] {
            background: var(--sk-panel) !important;
            border: 1px solid var(--sk-border) !important;
            color: var(--sk-text) !important;
        }
        .stButton button[data-testid="baseButton-secondary"]:hover,
        .stFormSubmitButton button[data-testid="baseButton-secondary"]:hover,
        .stButton button[kind="secondary"]:hover,
        .stFormSubmitButton button[kind="secondary"]:hover {
            border-color: var(--sk-focus) !important;
            color: var(--sk-text) !important;
        }
        .stButton button[data-testid="baseButton-primary"],
        .stFormSubmitButton button[data-testid="baseButton-primary"],
        .stButton button[kind="primary"],
        .stFormSubmitButton button[kind="primary"] {
            background: var(--sk-accent) !important;
            border: 1px solid var(--sk-accent) !important;
            color: #ffffff !important;
        }
        .stButton button[data-testid="baseButton-primary"]:hover,
        .stFormSubmitButton button[data-testid="baseButton-primary"]:hover,
        .stButton button[kind="primary"]:hover,
        .stFormSubmitButton button[kind="primary"]:hover {
            background: var(--sk-accent-hover) !important;
            border-color: var(--sk-accent-hover) !important;
            color: #ffffff !important;
        }
        div[role="radiogroup"] label {
            min-height: 48px;
            align-items: center;
            border-radius: 8px;
            color: var(--sk-text) !important;
        }
        input, textarea, select {
            min-height: 44px;
            font-size: 1rem !important;
            background: var(--sk-input) !important;
            color: var(--sk-text) !important;
            border-color: var(--sk-border) !important;
        }
        textarea {
            line-height: 1.45 !important;
        }
        *:focus-visible {
            outline: 3px solid var(--sk-focus) !important;
            outline-offset: 2px !important;
        }
        .sk-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
        }
        .sk-badge {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0.1rem 0.55rem;
            border: 1px solid;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .sk-badge-good {
            color: var(--sk-good-text);
            background: var(--sk-good-bg);
            border-color: var(--sk-good-border);
        }
        .sk-badge-warn {
            color: var(--sk-warn-text);
            background: var(--sk-warn-bg);
            border-color: var(--sk-warn-border);
        }
        .sk-badge-bad {
            color: var(--sk-bad-text);
            background: var(--sk-bad-bg);
            border-color: var(--sk-bad-border);
        }
        .sk-badge-info {
            color: var(--sk-info-text);
            background: var(--sk-info-bg);
            border-color: var(--sk-info-border);
        }
        .sk-badge-neutral {
            color: var(--sk-neutral-text);
            background: var(--sk-neutral-bg);
            border-color: var(--sk-neutral-border);
        }
        .sk-stage {
            border: 1px solid var(--sk-border);
            border-radius: 8px;
            background: var(--sk-panel);
            padding: 0.9rem 1rem;
            min-height: 116px;
            box-shadow: 0 8px 28px var(--sk-shadow);
        }
        .sk-stage-title {
            margin-top: 0.5rem;
            font-weight: 800;
            color: var(--sk-text);
        }
        .sk-stage-detail {
            margin-top: 0.35rem;
            color: var(--sk-muted);
            line-height: 1.45;
        }
        .sk-command {
            color: var(--sk-muted);
            overflow-wrap: anywhere;
        }
        </style>
        """
    css = css.replace("__SK_TOKENS__", _theme_tokens(appearance)).replace(
        "__SK_CLOSE__",
        close_auto_media,
    )
    st.markdown(css, unsafe_allow_html=True)


def _render_sidebar(
    repo_root: Path,
    state: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> str:
    with st.sidebar:
        _sync_pending_page()

        page = st.radio(
            "Primary task",
            PAGES,
            index=_safe_index(PAGES, st.session_state.get(PAGE_STATE_KEY)),
            key=PAGE_STATE_KEY,
            captions=[
                "Status",
                "Launch",
                "Jobs",
                "Files",
                "Commands",
            ],
        )

        if st.button(
            "Refresh",
            icon=":material/refresh:",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()

        active_count = len(_active_jobs(jobs))
        st.metric("Active jobs", active_count)
        st.metric("Runs", state["summary"]["run_count"])
        st.metric("Checkpoints", state["summary"]["checkpoint_count"])
        st.caption(str(repo_root))

    return str(page)


def _render_header(repo_root: Path, state: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    active_count = len(_active_jobs(jobs))
    status = "Active" if active_count else "Idle"
    tone = "info" if active_count else "good"

    left, right = st.columns([0.68, 0.32], vertical_alignment="center")
    with left:
        st.title("Project Skull")
    with right:
        st.markdown(
            "<div class='sk-row' style='justify-content:flex-end;'>"
            + _badge(status, tone)
            + _badge(f"{state['summary']['config_count']} configs", "neutral")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.caption(str(repo_root))


def _render_home(
    state: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> None:
    summary = state["summary"]
    ready_count = sum(1 for stage in state["pipeline"] if stage["status"] == "ready")
    stage_count = len(state["pipeline"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("Readiness", f"{ready_count}/{stage_count}")
    metric_columns[1].metric("Train configs", summary["train_config_count"])
    metric_columns[2].metric("Runs", summary["run_count"])
    metric_columns[3].metric("Active jobs", len(_active_jobs(jobs)))

    action_columns = st.columns(3)
    with action_columns[0]:
        if st.button(
            "Start a job",
            type="primary",
            icon=":material/play_arrow:",
            use_container_width=True,
        ):
            _set_page("Run")
            st.rerun()
    with action_columns[1]:
        if st.button(
            "Monitor work",
            icon=":material/monitoring:",
            use_container_width=True,
        ):
            _set_page("Monitor")
            st.rerun()
    with action_columns[2]:
        if st.button(
            "Review assets",
            icon=":material/folder_open:",
            use_container_width=True,
        ):
            _set_page("Assets")
            st.rerun()

    st.subheader("Pipeline")
    columns = st.columns(3)
    for index, stage in enumerate(state["pipeline"]):
        tone = {
            "ready": "good",
            "partial": "warn",
            "missing": "bad",
        }.get(stage["status"], "neutral")
        with columns[index % 3]:
            st.markdown(
                "<div class='sk-stage'>"
                + _badge(str(stage["status"]).title(), tone)
                + f"<div class='sk-stage-title'>{html.escape(stage['name'])}</div>"
                + f"<div class='sk-stage-detail'>{html.escape(stage['detail'])}</div>"
                + "</div>",
                unsafe_allow_html=True,
            )

    if jobs:
        st.subheader("Recent Jobs")
        _table(
            [
                {
                    "label": job.get("label"),
                    "status": job.get("status"),
                    "type": job.get("job_type"),
                    "created": job.get("created_at"),
                }
                for job in jobs[:5]
            ],
            height=220,
        )


def _render_train_config_summary(repo_root: Path, config: dict[str, Any]) -> None:
    missing = _missing_train_config_paths(repo_root, config)
    status = "Ready" if not missing else f"{len(missing)} missing"
    tone = "good" if not missing else "bad"

    st.markdown(
        "<div class='sk-row'>"
        + _badge(status, tone)
        + _badge(f"run: {_format_value(config.get('run_name'))}", "neutral")
        + _badge(f"steps: {_format_value(config.get('max_steps'))}", "neutral")
        + "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Device", _format_value(config.get("device")))
    cols[1].metric("Batch", _format_value(config.get("batch_size")))
    cols[2].metric("Block", _format_value(config.get("block_size")))
    cols[3].metric("Sources", _format_value(config.get("train_source_count")))

    if missing:
        _table(missing, height=180)


def _render_train_launcher(state: dict[str, Any], repo_root: Path) -> None:
    configs = _train_configs(state)
    if not configs:
        st.warning("No training configs were found.")
        return

    config_paths = [config["relative_path"] for config in configs]
    selected_config_path = st.selectbox("Training config", config_paths)
    selected_config = _train_config_record(state, selected_config_path)
    if selected_config:
        _render_train_config_summary(repo_root, selected_config)

    with st.form("train-launch-form"):
        mode_label = st.radio(
            "Mode",
            ["Auto", "Pretrain", "CPT", "SFT"],
            horizontal=True,
            captions=["Detect", "Base", "Continue", "Instruction"],
        )
        use_accelerate = st.checkbox("Use Accelerate", value=False)
        num_processes = st.number_input(
            "Processes",
            min_value=1,
            value=1,
            step=1,
            disabled=not use_accelerate,
        )
        default_label = f"train:{Path(selected_config_path).stem}"
        job_label = st.text_input("Job label", value=default_label)
        submitted = st.form_submit_button(
            "Start training",
            type="primary",
            icon=":material/play_arrow:",
            use_container_width=True,
        )

    if not submitted:
        return

    missing = _missing_train_config_paths(repo_root, selected_config)
    if missing:
        st.error("Training is blocked by missing config references.")
        _table(missing, height=220)
        return

    requested_mode = mode_label.lower()
    mode = _infer_train_mode(
        selected_config,
        selected_config_path,
        "auto" if requested_mode == "auto" else requested_mode,
    )
    command = build_train_command(
        mode,
        selected_config_path,
        use_accelerate=use_accelerate,
        num_processes=int(num_processes) if use_accelerate else None,
    )
    _start_and_focus(
        repo_root,
        job_type=f"train:{mode}",
        label=job_label,
        command=command,
        metadata={"config": selected_config_path, "mode": mode},
    )


def _render_eval_launcher(state: dict[str, Any], repo_root: Path) -> None:
    configs = _eval_configs(state)
    checkpoints = _checkpoint_paths(state)

    if not configs:
        st.warning("No eval configs were found.")
        return
    if not checkpoints:
        st.warning("No checkpoints were found.")
        return

    eval_paths = [config["relative_path"] for config in configs]
    with st.form("eval-launch-form"):
        config_path = st.selectbox("Eval config", eval_paths)
        checkpoint_path = st.selectbox("Checkpoint", checkpoints)
        print_json = st.checkbox("Print JSON", value=True)
        job_label = st.text_input(
            "Job label",
            value=f"eval:{Path(checkpoint_path).stem}",
        )
        submitted = st.form_submit_button(
            "Start evaluation",
            type="primary",
            icon=":material/check_circle:",
            use_container_width=True,
        )

    if not submitted:
        return
    if not _path_exists(repo_root, config_path, "Eval config"):
        return
    if not _path_exists(repo_root, checkpoint_path, "Checkpoint"):
        return

    _start_and_focus(
        repo_root,
        job_type="eval",
        label=job_label,
        command=build_eval_command(
            config_path,
            checkpoint_path,
            print_json=print_json,
        ),
        metadata={"config": config_path, "checkpoint": checkpoint_path},
    )


def _render_sample_launcher(state: dict[str, Any], repo_root: Path) -> None:
    train_configs = _train_configs(state)
    checkpoints = _checkpoint_paths(state)

    if not train_configs:
        st.warning("No training configs were found.")
        return
    if not checkpoints:
        st.warning("No checkpoints were found.")
        return

    config_paths = [config["relative_path"] for config in train_configs]
    with st.form("sample-launch-form"):
        config_path = st.selectbox("Config", config_paths)
        checkpoint_path = st.selectbox("Checkpoint", checkpoints)
        prompt = st.text_area("Prompt", value="Hello", height=120)
        max_new_tokens = st.slider("Max new tokens", 16, 512, 128, step=16)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, step=0.1)
        use_top_k = st.checkbox("Limit top-k", value=False)
        top_k = st.number_input(
            "Top-k",
            min_value=1,
            value=50,
            step=1,
            disabled=not use_top_k,
        )
        job_label = st.text_input(
            "Job label",
            value=f"sample:{Path(checkpoint_path).stem}",
        )
        submitted = st.form_submit_button(
            "Start sample",
            type="primary",
            icon=":material/auto_awesome:",
            use_container_width=True,
        )

    if not submitted:
        return
    if not prompt.strip():
        st.error("Prompt cannot be empty.")
        return
    if not _path_exists(repo_root, config_path, "Config"):
        return
    if not _path_exists(repo_root, checkpoint_path, "Checkpoint"):
        return

    _start_and_focus(
        repo_root,
        job_type="sample",
        label=job_label,
        command=build_sample_command(
            config_path,
            checkpoint_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=int(top_k) if use_top_k else None,
        ),
        metadata={"config": config_path, "checkpoint": checkpoint_path},
    )


def _render_test_launcher(repo_root: Path) -> None:
    with st.form("test-launch-form"):
        targets = st.text_area("Targets", value="tests", height=90)
        with st.expander("Advanced"):
            extra_args = st.text_input("Extra pytest args", value="")
        job_label = st.text_input("Job label", value="tests:pytest")
        submitted = st.form_submit_button(
            "Run tests",
            type="primary",
            icon=":material/science:",
            use_container_width=True,
        )

    if not submitted:
        return

    try:
        parsed_extra_args = _split_cli_args(extra_args)
    except ValueError as exc:
        st.error(f"Could not parse pytest args: {exc}")
        return

    _start_and_focus(
        repo_root,
        job_type="test",
        label=job_label,
        command=build_pytest_command(_nonempty_lines(targets), parsed_extra_args),
        metadata={"targets": _nonempty_lines(targets)},
    )


def _render_run(state: dict[str, Any], repo_root: Path) -> None:
    st.subheader("Run")
    action = st.radio(
        "Action",
        ["Train", "Evaluate", "Sample", "Tests"],
        horizontal=True,
        captions=["Fit model", "Score", "Generate", "Verify"],
    )

    if action == "Train":
        _render_train_launcher(state, repo_root)
    elif action == "Evaluate":
        _render_eval_launcher(state, repo_root)
    elif action == "Sample":
        _render_sample_launcher(state, repo_root)
    else:
        _render_test_launcher(repo_root)


def _render_job_status(job: dict[str, Any]) -> None:
    status = str(job.get("status", "unknown"))
    st.markdown(
        "<div class='sk-row'>"
        + _badge(status, _status_tone(status))
        + _badge(str(job.get("job_type", "job")), "neutral")
        + f"<span>{html.escape(str(job.get('label', 'Untitled')))}</span>"
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_jobs(repo_root: Path, jobs: list[dict[str, Any]]) -> None:
    st.subheader("Jobs")
    if not jobs:
        st.info("No web-launched jobs yet.")
        return

    active = _active_jobs(jobs)
    if active:
        with st.status(f"{len(active)} active job(s)", state="running", expanded=True):
            for job in active[:4]:
                st.write(
                    f"{job.get('label')} | {job.get('status')} | {job.get('created_at')}"
                )

    selected_job_id = st.selectbox(
        "Job",
        [job["id"] for job in jobs],
        format_func=lambda job_id: next(
            (
                f"{job.get('label')} ({job.get('status')})"
                for job in jobs
                if job.get("id") == job_id
            ),
            str(job_id),
        ),
    )
    job = next(item for item in jobs if item["id"] == selected_job_id)

    _render_job_status(job)
    cols = st.columns(4)
    cols[0].metric("Started", _format_value(job.get("started_at")))
    cols[1].metric("Finished", _format_value(job.get("finished_at")))
    cols[2].metric("Return code", _format_value(job.get("returncode")))
    cols[3].metric("PID", _format_value(job.get("child_pid") or job.get("runner_pid")))

    action_cols = st.columns(3)
    if job.get("status") in ACTIVE_JOB_STATUSES:
        with action_cols[0]:
            if st.button(
                "Stop job",
                type="primary",
                icon=":material/stop_circle:",
                use_container_width=True,
            ):
                request_stop(repo_root, selected_job_id)
                _set_notice("warning", f"Stop requested for {job.get('label')}.")
                st.cache_data.clear()
                st.rerun()
    else:
        with action_cols[0]:
            remove_log = st.checkbox("Delete log too", value=False)
        with action_cols[1]:
            if st.button(
                "Delete job",
                icon=":material/delete:",
                use_container_width=True,
            ):
                try:
                    delete_job(repo_root, selected_job_id, delete_log=remove_log)
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    _set_notice("success", "Job deleted.")
                    st.cache_data.clear()
                    st.rerun()

    st.markdown(
        f"<div class='sk-command'>{html.escape(str(job.get('display_command', '')))}</div>",
        unsafe_allow_html=True,
    )
    st.text_area(
        "Log",
        value=read_log_tail(job, max_chars=12000),
        height=320,
        disabled=True,
    )


def _run_progress(
    state: dict[str, Any],
    run: dict[str, Any],
) -> tuple[float | None, float | None]:
    config = _match_run_config(state, run)
    latest_train = run.get("latest_train", {})
    max_steps = config.get("max_steps") if config else None
    if not isinstance(max_steps, int | float) or not max_steps:
        return None, None

    progress = min(max(run.get("latest_step", 0) / float(max_steps), 0.0), 1.0)
    steps_per_sec = latest_train.get("steps_per_sec")
    if (
        isinstance(steps_per_sec, int | float)
        and steps_per_sec > 0
        and run.get("latest_step", 0) < max_steps
    ):
        eta = (float(max_steps) - float(run.get("latest_step", 0))) / float(
            steps_per_sec
        )
        return progress, eta
    return progress, None


def _render_metrics_charts(run: dict[str, Any]) -> None:
    metrics = run.get("metrics_rows", [])
    if not metrics:
        st.info("No metrics were found for this run.")
        return

    metrics_df = pd.DataFrame(metrics)
    if "step" not in metrics_df.columns:
        st.info("Metrics were found, but no step column was present.")
        return

    metrics_df = metrics_df.sort_values("step")
    chart_specs = [
        ("Loss", ["train_loss", "val_loss"]),
        ("Accuracy", ["train_acc", "val_acc"]),
        ("Learning rate", ["lr"]),
        ("Throughput", ["tokens_per_sec", "steps_per_sec"]),
        ("Gradient norm", ["grad_norm"]),
    ]
    visible_specs = [
        (title, columns)
        for title, columns in chart_specs
        if any(column in metrics_df.columns for column in columns)
    ]

    if not visible_specs:
        _table(metrics_df.to_dict("records"), height=260)
        return

    for start in range(0, len(visible_specs), 2):
        row_specs = visible_specs[start : start + 2]
        columns = st.columns(len(row_specs))
        for column, (title, metric_columns) in zip(columns, row_specs, strict=False):
            visible_columns = [
                value for value in ["step", *metric_columns] if value in metrics_df
            ]
            chart_df = metrics_df[visible_columns].copy()
            subset = [value for value in visible_columns if value != "step"]
            if subset:
                chart_df = chart_df.dropna(how="all", subset=subset)
            with column:
                st.caption(title)
                if chart_df.empty:
                    st.info("No values.")
                else:
                    st.line_chart(chart_df.set_index("step"))


def _render_runs(state: dict[str, Any]) -> None:
    st.subheader("Runs")
    runs = state["runs"]
    if not runs:
        st.info("No run directories with metrics or checkpoints were found.")
        return

    selected_run_path = st.selectbox(
        "Run",
        [run["relative_path"] for run in runs],
    )
    run = next(item for item in runs if item["relative_path"] == selected_run_path)
    latest_train = run.get("latest_train", {})
    latest_val = run.get("latest_val", {})
    progress, eta = _run_progress(state, run)

    columns = st.columns(5)
    columns[0].metric("Step", _format_value(run.get("latest_step")))
    columns[1].metric("Train loss", _format_value(latest_train.get("train_loss")))
    columns[2].metric("Val loss", _format_value(latest_val.get("val_loss")))
    columns[3].metric("Best val", _format_value(run.get("best_val_loss")))
    columns[4].metric("ETA", _format_duration(eta))

    if progress is not None:
        st.progress(progress)

    _render_metrics_charts(run)

    detail = st.radio(
        "Run detail",
        ["Checkpoints", "Errors", "Samples"],
        horizontal=True,
    )
    if detail == "Checkpoints":
        _table(
            [
                {
                    "name": item["name"],
                    "path": item["relative_path"],
                    "size": item["size_human"],
                    "modified": item["modified_at"],
                }
                for item in run.get("checkpoints", [])
            ],
            height=240,
        )
    elif detail == "Errors":
        _table(
            [
                {
                    "step": item.get("step"),
                    "stage": item.get("stage"),
                    "action": item.get("action"),
                    "type": item.get("error_type"),
                    "message": item.get("message"),
                }
                for item in run.get("errors_rows", [])
            ],
            height=240,
        )
    else:
        samples = run.get("samples", [])
        if not samples:
            st.info("No samples were found for this run.")
            return
        selected_sample = st.selectbox(
            "Sample",
            [sample["name"] for sample in samples],
        )
        sample = next(item for item in samples if item["name"] == selected_sample)
        st.text_area(
            "Sample text",
            value=str(sample.get("preview", "")),
            height=220,
            disabled=True,
        )


def _render_monitor(
    state: dict[str, Any],
    repo_root: Path,
    jobs: list[dict[str, Any]],
) -> None:
    mode = st.radio("Monitor", ["Jobs", "Runs"], horizontal=True)
    if mode == "Jobs":
        _render_jobs(repo_root, jobs)
    else:
        _render_runs(state)


def _render_pipeline_table(state: dict[str, Any]) -> None:
    _table(
        [
            {
                "stage": stage["name"],
                "status": stage["status"],
                "detail": stage["detail"],
            }
            for stage in state["pipeline"]
        ],
        height=260,
    )


def _render_configs(state: dict[str, Any]) -> None:
    configs = state["configs"]
    if not configs:
        st.info("No YAML configs were found.")
        return

    kind = st.radio(
        "Config group",
        ["All", "Train", "Model", "Eval"],
        horizontal=True,
    )
    visible = [
        item
        for item in configs
        if kind == "All" or item.get("kind") == kind.lower()
    ]
    _table(
        [
            {
                "path": item.get("relative_path"),
                "kind": item.get("kind"),
                "run": item.get("run_name"),
                "device": item.get("device"),
                "steps": item.get("max_steps"),
                "block": item.get("block_size"),
            }
            for item in visible
        ],
        height=280,
    )
    if not visible:
        return

    selected_path = st.selectbox(
        "Inspect",
        [item["relative_path"] for item in visible],
    )
    selected = next(item for item in visible if item["relative_path"] == selected_path)
    st.json(selected.get("content", {}), expanded=1)


def _render_data_assets(state: dict[str, Any]) -> None:
    assets = state["data_assets"]
    group = st.radio(
        "Data group",
        ["Tokenizers", "Clean corpora", "Bin shards", "Manifests"],
        horizontal=True,
    )

    if group == "Tokenizers":
        rows = [
            {
                "name": item["name"],
                "path": item["relative_path"],
                "size": item["size_human"],
                "modified": item["modified_at"],
            }
            for item in assets["tokenizers"]
        ]
    elif group == "Clean corpora":
        rows = [
            {
                "name": item["name"],
                "path": item["relative_path"],
                "size": item["size_human"],
                "modified": item["modified_at"],
            }
            for item in assets["clean_files"]
        ]
    elif group == "Bin shards":
        rows = [
            {
                "name": item["name"],
                "path": item["relative_path"],
                "train": item["train_shards"],
                "val": item["val_shards"],
                "tokens": item["meta_tokens"],
                "size": item["total_size_human"],
            }
            for item in assets["bins"]
        ]
    else:
        rows = [
            {
                "name": item["name"],
                "path": item["relative_path"],
                "size": item["size_human"],
                "modified": item["modified_at"],
            }
            for item in assets["manifests"]
        ]

    _table(rows, height=300)


def _render_scripts(state: dict[str, Any]) -> None:
    _table(
        [
            {
                "script": script["relative_path"],
                "args": " ".join(script.get("arguments", [])),
                "modified": script["modified_at"],
            }
            for script in state["scripts"]
        ],
        height=320,
    )


def _render_assets(state: dict[str, Any]) -> None:
    st.subheader("Assets")
    area = st.radio(
        "Area",
        ["Pipeline", "Configs", "Data", "Scripts"],
        horizontal=True,
    )
    if area == "Pipeline":
        _render_pipeline_table(state)
    elif area == "Configs":
        _render_configs(state)
    elif area == "Data":
        _render_data_assets(state)
    else:
        _render_scripts(state)


def _render_help() -> None:
    st.subheader("Commands")
    st.code(
        "pip install -e .[dev,web]\npython -m skull.web",
        language="bash",
    )
    st.code(
        "python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml\n"
        "python -m skull.cli.cpt --config configs/train/cpt_150m.yaml\n"
        "python -m skull.cli.sft --config configs/train/sft_150m.yaml",
        language="bash",
    )
    st.code(
        "python -m skull.cli.eval --config configs/eval/default_eval.yaml "
        "--ckpt runs/pretrain/skull_150m_base/best.pt\n"
        "python -m skull.cli.sample --config configs/train/pretrain_150m.yaml "
        "--ckpt runs/pretrain/skull_150m_base/best.pt --prompt \"Hello\"",
        language="bash",
    )


def _render_page(
    page: str,
    state: dict[str, Any],
    repo_root: Path,
    jobs: list[dict[str, Any]],
) -> None:
    if page == "Home":
        _render_home(state, jobs)
    elif page == "Run":
        _render_run(state, repo_root)
    elif page == "Monitor":
        _render_monitor(state, repo_root, jobs)
    elif page == "Assets":
        _render_assets(state)
    else:
        _render_help()


def main() -> None:
    st.set_page_config(
        page_title="Project Skull",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.markdown("### Project Skull")
    appearance = st.sidebar.radio(
        "Appearance",
        ["Auto", "Light", "Dark"],
        horizontal=True,
        key="appearance",
    )
    _inject_styles(appearance)

    repo_root = Path(
        st.sidebar.text_input("Repo root", value=str(DEFAULT_REPO_ROOT))
    ).expanduser().resolve()
    if not repo_root.exists():
        st.error(f"Repo root does not exist: {repo_root}")
        return

    state = _load_state(str(repo_root))
    jobs = load_jobs(repo_root)
    page = _render_sidebar(repo_root, state, jobs)

    _render_header(repo_root, state, jobs)
    _display_notice()
    _render_page(page, state, repo_root, jobs)


if __name__ == "__main__":
    main()
