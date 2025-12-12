import logging
from collections import deque


class BlenderUILogger:

    _max_msgs: int = 20

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Prevent double handlers when modules reload
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[SOAP] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.propagate = False

        self.info_buffer = deque()

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)
        self.info_buffer.append(("INFO", str(msg)))

    def warn(self, msg):
        self.info_buffer.append(("WARNING", str(msg)))

    def error(self, msg, exc: Exception = None):
        if exc is not None:
            self.logger.error(msg, exc_info=exc)
            self.info_buffer.append(("ERROR", f"{msg}: {exc}"))
        else:
            self.logger.error(msg)
            self.info_buffer.append(("ERROR", str(msg)))

    def coalesce(self, caller):
        count = 0
        while self.info_buffer and count < BlenderUILogger._max_msgs:
            lvl, msg = self.info_buffer.pop()
            for line in str(msg).split("\n"):
                caller.report({lvl}, line.strip())
            count += 1


LOGGER = BlenderUILogger("SoapTools")
