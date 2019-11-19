from base.base_dataloader import BaseDataLoader
import tensorflow as tf
from glob import glob
from functools import partial
import random


# TODO: Create seperate dataset for true positve and true ngatives
def _generator(valid_path, img_size):
    valid_files_src = glob(valid_path)
    random.shuffle(valid_files_src)
    valid_files_tar = glob(valid_path)
    random.shuffle(valid_files_tar)
    for src, tar in zip(valid_files_src, valid_files_tar):
        label_src = int(src.split("/")[-3])  # get the identification from path
        label_tar = int(tar.split("/")[-3])

        img_src = tf.io.read_file(src)
        img_src = tf.image.decode_jpeg(img_src, channels=3)
        img_src = tf.image.convert_image_dtype(img_src, tf.float32)
        img_src = tf.image.resize(img_src, [img_size, img_size])

        img_tar = tf.io.read_file(tar)
        img_tar = tf.image.decode_jpeg(img_tar, channels=3)
        img_tar = tf.image.convert_image_dtype(img_tar, tf.float32)
        img_tar = tf.image.resize(img_tar, [img_size, img_size])
        yield img_src, img_tar, label_src == label_tar


class Dataloader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.train_ds = None
        self.valid_ds = None  # valid dataset for true negative pairs
        self.train_size = len(glob(self.config.data.train_faces_path))
        self.valid_size = len(glob(self.config.data.valid_faces_path))

        # initialize datasets
        self.init_train_ds()
        self.init_valid_ds()

    @staticmethod
    def _process_path_train(file_path, depth, on_value=1.0, off_value=0.0):
        label_int = int(tf.strings.split(file_path, "/")[-2])
        label = tf.one_hot(indices=label_int, depth=depth, on_value=on_value, off_value=off_value)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    def init_train_ds(self):
        process_path_train = partial(self._process_path_train, depth=self.config.data.n_classes, on_value=1.0,
                                     off_value=0.0)
        self.train_ds = tf.data.Dataset.list_files(self.config.data.train_faces_path)
        self.train_ds = self.train_ds.shuffle(self.train_size)
        self.train_ds = self.train_ds.map(process_path_train, num_parallel_calls=self.config.data.n_workers)
        self.train_ds = self.train_ds.batch(batch_size=self.config.trainer.train_batch_size)
        self.train_ds = self.train_ds.prefetch(1)
        return

    def init_valid_ds(self):
        valid_generator = partial(_generator, valid_path=self.config.data.valid_faces_path,
                                  img_size=self.config.data.img_size)
        self.valid_ds = tf.data.Dataset.from_generator(generator=valid_generator,
                                                       output_types=(tf.float32, tf.float32, tf.bool))

        self.valid_ds = self.valid_ds.shuffle(self.valid_size)
        self.valid_ds = self.valid_ds.batch(batch_size=self.config.trainer.valid_batch_size)
        self.valid_ds = self.valid_ds.prefetch(1)
        return

    def get_train_data_generator(self):
        return self.train_ds

    def get_valid_data_generator(self):
        return self.valid_ds

    def get_train_data_size(self):
        return self.train_size

    def get_valid_data_size(self):
        return self.valid_size
