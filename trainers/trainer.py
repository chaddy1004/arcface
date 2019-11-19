import os
from collections import defaultdict

import tensorflow as tf

from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):

    def __init__(self, data_loader, config, backbone, full):
        super().__init__(data_loader, config)
        self.train_ds = data_loader.get_train_data_generator()
        self.valid_ds = data_loader.get_valid_data_generator()
        self.backbone = backbone
        self.full = full
        self.writer = tf.summary.create_file_writer(self.config.exp.log_dir)
        self.train_metric_logs = defaultdict()
        self.valid_metric_logs = defaultdict()

    @staticmethod
    def train_metric_names():
        return ["loss", "accuracy"]

    @staticmethod
    def valid_metric_names():
        return ["loss", "accuracy", "tp", "tn", "fp", "fn"]

    def train(self):
        for epoch in range(self.config.trainer.num_epochs):
            train_loss = 0
            train_acc = 0
            num_steps = 0
            train_metric_names = self.train_metric_names()
            for img_train, label_train in self.train_ds:
                metric_logs = {}
                loss = self.full.train_on_batch(img_train, label_train)
                assert len(loss) == len(train_metric_names)
                for metric_name, metric_val in zip(train_metric_names, loss):
                    metric_logs[f"train/{metric_name}"] = metric_val

                print(f"Epoch:{epoch} Step:{num_steps} train_loss: {loss[0]}, train_acc: {loss[1]}")
                with self.writer.as_default():
                    for name, value in metric_logs.items():
                        tf.summary.scalar(name, value, num_steps)
                num_steps += 1
            # valid_metric_names = self.valid_metric_names()

            if epoch + 1 % self.config.trainer.save_checkpoint_freq:
                filename = os.path.join(self.config.exp.saved_model_dir, f'model.hdf5')
                self.full.save(filename=filename, overwrite=True)

            # test_loss = 0
            # test_acc = 0
            # num_steps = 0
            # for img_src, img_tar, match in self.test_dataset:
            #     pass
            #     # img_src is the image that is being compared to target image (img_tar). Match is boolean on whether img_src and img_tar is saame person (match = True) or not (match = False)
            #     # If prediction for img_src says it is same as , it is tp (true positive), and so on
            #     # REQUIRE CUSTOM MATCH
            #     # loss = self.model.test_on_batch(image_test, label_test)
            #     # print(f"Epoch:{epoch} Step:{num_steps} test_loss: {loss[0]}, test_acc: {loss[1]}")
            #     # test_loss += loss[0]
            #     # test_acc += loss[1]
            #     # num_steps += 1
            # test_loss /= num_steps
            # test_acc /= num_steps
            # self.test_losses.append(test_loss)
            # self.test_accs.append(test_acc)

            # with self.writer.as_default():
            #     tf.summary.scalar("loss/train", train_loss, epoch)
            #     tf.summary.scalar("acc/train", train_acc, epoch)
            # tf.summary.scalar("loss/test", test_loss, epoch)
            # tf.summary.scalar("acc/test", test_acc, epoch)
