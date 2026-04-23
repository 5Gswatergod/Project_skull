from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web.cli import main as streamlit_main
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -e .[web]` first."
        ) from exc

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(streamlit_main())


if __name__ == "__main__":
    main()
