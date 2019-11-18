import abc


class BaseModel:
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def define_backbone(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def build_backbone(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def define_full(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def build_full(self, **kwargs):
        raise NotImplementedError


