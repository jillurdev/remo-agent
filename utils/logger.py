import json
import logging
import os
import time


class JsonFormatter(logging.Formatter):
    """Production format: one JSON object per line, easy to ship to log
    aggregators (Datadog, CloudWatch, etc)."""

    def format(self, record):
        log_record = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "room": getattr(record, "room", None),
            "event": getattr(record, "event", record.msg),
            "data": getattr(record, "data", {}),
        }
        if record.exc_info:
            log_record["error"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class PrettyFormatter(logging.Formatter):
    """Dev format: short, human-readable single line, easy to eyeball while
    testing locally (python main.py dev)."""

    LEVEL_COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        color = self.LEVEL_COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<7}{self.RESET}"
        msg = record.getMessage()
        line = f"{ts} {level} {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logger():
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        ch = logging.StreamHandler()
        # Default to the readable format for local development. Set
        # LOG_FORMAT=json in production (e.g. Railway) to switch back to
        # structured JSON logs for log aggregators.
        if os.getenv("LOG_FORMAT", "pretty").lower() == "json":
            ch.setFormatter(JsonFormatter())
        else:
            ch.setFormatter(PrettyFormatter())
        logger.addHandler(ch)

    return logger


logger = setup_logger()
