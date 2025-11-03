import threading


class BackgroundJob:
    def __init__(self, func, *args, **kwargs):
        self.result = None
        self.done = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._wrapper, args=(func, args, kwargs))
        self._thread.start()

    def _wrapper(self, func, args, kwargs):
        res = func(*args, **kwargs)
        with self._lock:
            self.result = res
            self.done = True

    def is_done(self):
        with self._lock:
            return self.done

    def get_result(self):
        with self._lock:
            return self.result
