import os
from collections import defaultdict

import tensorflow as tf
from utils.valid_helper import cos_similarity, face_match

from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):

    def __init__(self, data_loader, config, backbone, full):
        super().__init__(data_loader, config)
        self.train_ds = data_loader.get_train_data_generator()
        self.valid_ds = data_loader.get_valid_data_generator()
        # self.valid_tn_ds = data_loader.get_valid_tn_data_generator()
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
        return ["tp", "tn", "fp", "fn"]

    def train(self):
        for epoch in range(self.config.trainer.num_epochs):
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

            num_steps = 0
            for img_src, img_tar, gt in self.valid_ds:
                src_feature = self.backbone(img_src)
                tar_feature = self.backbone(img_tar)
                result = face_match(src_feature, tar_feature)
                result_list = list(result.numpy().squeeze())
                gt_list = list(gt.numpy().squeeze())
                result_logs = defaultdict(int)
                assert len(gt_list) == len(result_list)
                for true, result in zip(gt_list, result_list):
                    if true and result:
                        result_logs["tp"] += 1
                    elif not true and not result:
                        result_logs["tn"] += 1
                    elif not true and result:
                        result_logs["fp"] += 1
                    elif true and not result:
                        result_logs["fn"] += 1
                metric_logs = defaultdict(int)
                if sum(result_list) == 0:
                    metric_logs[f"valid/acc_tn"] = result_logs["tn"] / len(result_list)
                elif sum(result_list) == len(result_list):
                    metric_logs[f"valid/acc_tp"] = result_logs["tp"] / len(result_list)
                else:
                    metric_logs[f"valid/acc_tp"] = result_logs["tp"] / sum(result_list)
                    metric_logs[f"valid/acc_tn"] = result_logs["tn"] / (1 - sum(result_list))

                with self.writer.as_default():
                    for name, value in metric_logs.items():
                        tf.summary.scalar(name, value, num_steps)

                acc_tn = metric_logs[f"valid/acc_tn"]
                acc_tp = metric_logs[f"valid/acc_tp"]
                print(f"Epoch:{epoch} Step:{num_steps} acc_tn: {acc_tn}, acc_tp: {acc_tp}")
                num_steps += 1

            if epoch + 1 % self.config.trainer.save_checkpoint_freq:
                filename = os.path.join(self.config.exp.saved_model_dir, f'model.hdf5')
                self.full.save(filename, overwrite=True)
                print(f"Model saved to {filename}")
