from abc import ABC, abstractmethod


class Singleton(ABC):
    _instances = {}

    def __new__(cls, *args, **kwargs):

        if cls not in cls._instances:
            instance = super().__new__(cls)
            instance.initialize(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass
