import abc


class BaseTrainer(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
