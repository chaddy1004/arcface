import abc


class BaseModel:
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def define_backbone(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def compile_backbone(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def define_full(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def compile_full(self, **kwargs):
        raise NotImplementedError


