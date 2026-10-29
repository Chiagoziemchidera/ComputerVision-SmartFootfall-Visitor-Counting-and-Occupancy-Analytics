from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .application import configure_logging, run


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Multi-camera footfall and visitor analytics"
    )
    parser.add_argument(
        "--config",
        default="config/config.example.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    arguments = parser.parse_args()

    configure_logging(arguments.log_level)
    run(arguments.config)


if __name__ == "__main__":
    main()
