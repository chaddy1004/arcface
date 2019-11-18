from base.base_dataloader import BaseDataLoader
import tensorflow as tf
from glob import glob
from functools import partial


class Dataloader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.train_ds = None
        self.valid_ds = None
        self.train_size = len(glob(self.config.data.train_faces_path))
        self.valid_size = len(glob(self.config.data.valid_faces_path))

        # initialize datasets
        self.init_train_ds()
        self.init_valid_ds()

    @staticmethod
    def _process_path(file_path, depth, on_value=1.0, off_value=0.0):
        label_int = int(tf.strings.split(file_path, "/")[-2])
        label = tf.one_hot(indices=label_int, depth=depth, on_value=on_value, off_value=off_value)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    def init_train_ds(self):
        process_path = partial(self._process_path, depth=self.config.data.n_classes, on_value=1.0, off_value=0.0)
        self.train_ds = tf.data.Dataset.list_files(self.config.data.train_faces_path)
        self.train_ds = self.train_ds.shuffle(self.train_size)
        self.train_ds = self.train_ds.map(process_path, num_parallel_calls=self.config.data.n_workers)
        self.train_ds = self.train_ds.batch(batch_size=self.config.trainer.batch_size)
        self.train_ds = self.train_ds.prefetch(1)
        return

    def init_valid_ds(self):
        pass

    def get_train_data_generator(self):
        return self.train_ds

    def get_valid_data_generator(self):
        return self.valid_ds

    def get_train_data_size(self):
        return self.train_size

    def get_valid_data_size(self):
        return self.valid_size
