from __future__ import annotations

from stair_monitor.config.settings import SETTINGS, load_camera_config
from stair_monitor.output.logging_setup import setup_app_logging
from stair_monitor.runtime.runtime_factory import create_runtime_app


def main() -> None:
    setup_app_logging(SETTINGS.logging.log_dir)
    behavior_config = load_camera_config()
    app = create_runtime_app(
        settings=SETTINGS,
        behavior_config=behavior_config,
    )
    app.run()


if __name__ == "__main__":
    main()
