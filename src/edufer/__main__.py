from __future__ import annotations

import uvicorn

from edufer.core.settings import DEFAULT_CONFIG_PATH, AppSettings


def main() -> None:
    settings = AppSettings.from_file(DEFAULT_CONFIG_PATH)
    uvicorn.run(
        "edufer.app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
