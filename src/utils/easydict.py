class EasyDict(dict):
    """A minimal EasyDict implementation with full recursive dot-access."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._convert_value(v)

    def _convert_value(self, v):
        # Convert dicts recursively
        if isinstance(v, dict):
            return EasyDict(v)
        # Convert dicts inside lists or tuples recursively
        elif isinstance(v, list):
            return [self._convert_value(x) for x in v]
        elif isinstance(v, tuple):
            return tuple(self._convert_value(x) for x in v)
        # leave everything else as is
        else:
            return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"'EasyDict' object has no attribute '{name}'") from e

    def __setattr__(self, name, value):
        self[name] = self._convert_value(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"'EasyDict' object has no attribute '{name}'") from e
