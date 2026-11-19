import logging
import os


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or os.getenv("ANPR_LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
