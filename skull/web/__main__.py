from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise SystemExit(
            "Web server dependencies are not installed. Run `pip install -e .[web]` first."
        ) from exc

    default_repo_root = Path(
        os.environ.get("SKULL_REPO_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()

    parser = argparse.ArgumentParser(description="Project Skull web app")
    parser.add_argument("--host", default=os.environ.get("SKULL_WEB_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SKULL_WEB_PORT", "8000")),
    )
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--repo-root",
        default=str(default_repo_root),
        help="Repository root to inspect by default.",
    )
    args = parser.parse_args(sys.argv[1:])

    os.environ["SKULL_REPO_ROOT"] = str(Path(args.repo_root).expanduser().resolve())
    uvicorn.run(
        "skull.web.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
