import json
import logging
import time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "room": getattr(record, "room", None),
            "event": getattr(record, "event", record.msg),
            "data": getattr(record, "data", {})
        }
        if record.exc_info:
            log_record["error"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logger():
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(JsonFormatter())
        logger.addHandler(ch)
    
    return logger

logger = setup_logger()
