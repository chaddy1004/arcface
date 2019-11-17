from base.base_dataloader import BaseDataLoader


class Dataloader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

        self.train_dataset = None
        self.valid_dataset = None
        self.train_size = 0
        self.valid_size = 0

    def init_train_dataset(self):

        pass

    def init_valid_dataset(self):
        pass

