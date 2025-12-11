import threading

from typing import Optional, Any, Tuple


class BackgroundJob:
    def __init__(self, func, *args, **kwargs):
        self.result = None
        self.done = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._wrapper, args=(func, args, kwargs))
        self._thread.start()

    def _wrapper(self, func, args, kwargs):
        self.result = self.exception = None
        try:
            res = func(*args, **kwargs)
            with self._lock:
                self.result = res
        except Exception as e:
            with self._lock:
                self.exception = e
        finally:
            with self._lock:
                self.done = True

    def is_done(self) -> bool:
        with self._lock:
            return self.done

    def get_result(self) -> Tuple[Optional[Any], Optional[Exception]]:
        with self._lock:
            return self.result, self.exception
