import logging
import json
import os
from datetime import datetime, timezone


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        event = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname, "logger": record.name, "message": record.getMessage(),
        }
        context = getattr(record, "anpr_metrics", None)
        if context is not None:
            event["metrics"] = context
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False)


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or os.getenv("ANPR_LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if os.getenv("ANPR_LOG_FORMAT") == "json":
        for handler in logging.getLogger().handlers:
            handler.setFormatter(JsonLogFormatter())
