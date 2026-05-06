from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from skull.web.service import (
    build_dashboard_payload,
    default_repo_root,
    delete_finished_job,
    get_job_log,
    get_job_or_raise,
    launch_eval_job,
    launch_sample_job,
    launch_test_job,
    launch_train_job,
    resolve_repo_root,
    stop_job,
)


class RepoRootBody(BaseModel):
    repo_root: str | None = None


class TrainLaunchRequest(RepoRootBody):
    config_path: str
    requested_mode: str = "auto"
    use_accelerate: bool = False
    num_processes: int | None = None
    label: str | None = None


class EvalLaunchRequest(RepoRootBody):
    config_path: str
    checkpoint_path: str
    print_json: bool = True
    label: str | None = None


class SampleLaunchRequest(RepoRootBody):
    config_path: str
    checkpoint_path: str
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    label: str | None = None


class TestLaunchRequest(RepoRootBody):
    targets: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    label: str | None = None


class DeleteJobRequest(RepoRootBody):
    delete_log_too: bool = False


def _handle_service_error(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


def _frontend_ready(static_dir: Path) -> bool:
    return (static_dir / "index.html").exists()


def _frontend_placeholder(static_dir: Path) -> HTMLResponse:
    root = default_repo_root()
    message = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Project Skull</title>
        <style>
          body {{
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            background: #0b1018;
            color: #f8fafc;
            font-family: ui-sans-serif, system-ui, sans-serif;
          }}
          main {{
            width: min(720px, calc(100vw - 48px));
            padding: 32px;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 25px 80px rgba(0, 0, 0, 0.35);
          }}
          code {{
            padding: 0.12rem 0.4rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.2);
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>Frontend bundle not found</h1>
          <p>The API server is running, but <code>{static_dir.as_posix()}</code> does not contain a built React app yet.</p>
          <p>Build the UI from the repo root with <code>npm install --prefix frontend</code> and <code>npm run build --prefix frontend</code>, then reload this page.</p>
          <p>Default repo root: <code>{root.as_posix()}</code></p>
        </main>
      </body>
    </html>
    """
    return HTMLResponse(message, status_code=503)


def _static_file(static_dir: Path, full_path: str) -> Path | None:
    requested = full_path.lstrip("/")
    if not requested:
        return static_dir / "index.html"

    candidate = (static_dir / requested).resolve()
    try:
        candidate.relative_to(static_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def create_app() -> FastAPI:
    app = FastAPI(title="Project Skull", version="0.1.0")
    static_dir = Path(__file__).with_name("static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(repo_root: str | None = Query(default=None)) -> dict[str, object]:
        root = resolve_repo_root(repo_root)
        return {
            "status": "ok",
            "repo_root": str(root),
            "frontend_ready": _frontend_ready(static_dir),
        }

    @app.get("/api/dashboard")
    def dashboard(repo_root: str | None = Query(default=None)) -> dict[str, object]:
        root = resolve_repo_root(repo_root)
        return build_dashboard_payload(root)

    @app.get("/api/jobs/{job_id}")
    def job_detail(
        job_id: str,
        repo_root: str | None = Query(default=None),
    ) -> dict[str, object]:
        root = resolve_repo_root(repo_root)
        try:
            return {"job": get_job_or_raise(root, job_id)}
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(
        job_id: str,
        repo_root: str | None = Query(default=None),
        max_chars: int = Query(default=20000, ge=1000, le=100000),
    ) -> dict[str, object]:
        root = resolve_repo_root(repo_root)
        try:
            return get_job_log(root, job_id, max_chars=max_chars)
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.post("/api/launch/train")
    def launch_train(payload: TrainLaunchRequest) -> dict[str, object]:
        root = resolve_repo_root(payload.repo_root)
        try:
            job = launch_train_job(
                root,
                config_path=payload.config_path,
                requested_mode=payload.requested_mode,  # type: ignore[arg-type]
                use_accelerate=payload.use_accelerate,
                num_processes=payload.num_processes,
                label=payload.label,
            )
            return {"job": job}
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.post("/api/launch/eval")
    def launch_eval(payload: EvalLaunchRequest) -> dict[str, object]:
        root = resolve_repo_root(payload.repo_root)
        try:
            job = launch_eval_job(
                root,
                config_path=payload.config_path,
                checkpoint_path=payload.checkpoint_path,
                print_json=payload.print_json,
                label=payload.label,
            )
            return {"job": job}
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.post("/api/launch/sample")
    def launch_sample(payload: SampleLaunchRequest) -> dict[str, object]:
        root = resolve_repo_root(payload.repo_root)
        try:
            job = launch_sample_job(
                root,
                config_path=payload.config_path,
                checkpoint_path=payload.checkpoint_path,
                prompt=payload.prompt,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                label=payload.label,
            )
            return {"job": job}
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.post("/api/launch/test")
    def launch_test(payload: TestLaunchRequest) -> dict[str, object]:
        root = resolve_repo_root(payload.repo_root)
        try:
            job = launch_test_job(
                root,
                targets=payload.targets,
                extra_args=payload.extra_args,
                label=payload.label,
            )
            return {"job": job}
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.post("/api/jobs/{job_id}/stop")
    def stop(job_id: str, payload: RepoRootBody) -> dict[str, object]:
        root = resolve_repo_root(payload.repo_root)
        try:
            return {"job": stop_job(root, job_id)}
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.delete("/api/jobs/{job_id}")
    def remove_job(job_id: str, payload: DeleteJobRequest) -> dict[str, object]:
        root = resolve_repo_root(payload.repo_root)
        try:
            return {
                "job": delete_finished_job(
                    root,
                    job_id,
                    delete_log_too=payload.delete_log_too,
                )
            }
        except Exception as exc:  # pragma: no cover - thin HTTP wrapper
            _handle_service_error(exc)

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str = ""):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if not _frontend_ready(static_dir):
            return _frontend_placeholder(static_dir)
        file_path = _static_file(static_dir, full_path)
        if file_path is not None:
            return FileResponse(file_path)
        return FileResponse(static_dir / "index.html")

    return app
