import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, Activation, Lambda, \
    GlobalAveragePooling2D

from base.base_model import BaseModel
from utils.layers import L2Normalize
from utils.losses import _arcface
from functools import partial
from tensorflow.keras.optimizers import Adam


class Res50(BaseModel):
    # Backbone is used for valid and prediction and only outptus 512-sized feature embedding
    def define_backbone(self, model_name: str):
        image = Input(shape=self.config.data.img_shape, name="img_in_backbone")
        res50 = tf.keras.applications.ResNet50(include_top=False, input_shape=(None, None, 3))
        x = res50(image)
        # x = Dense(1000)(image) # for testing the error
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512)(x)
        feature = BatchNormalization()(x)
        feature = Lambda(lambda _x: tf.math.l2_normalize(_x))(feature)  # normalizing the feature vector
        backbone = Model(inputs=image, outputs=feature, name=model_name)
        return backbone

    def build_backbone(self, model_name):
        # no explicit training of just backbone, so no need to compile the model. Hence, no implementation is needed
        pass

    # full model is the one with softmax at the end for meta learning stage. It is made with the backbone
    def define_full(self, backbone, model_name):
        image = Input(shape=self.config.data.img_shape, name="img_in_full")
        feature = backbone(image)
        logit_pre = L2Normalize(Dense(units=self.config.data.n_classes))(feature)  # dim: n_class
        # logit_pre = Dense(units=self.config.data.n_classes)(feature)
        # logit_pre = Activation("softmax")(logit_pre)
        full_model = Model(inputs=image, outputs=logit_pre, name=model_name)
        return full_model

    def build_full(self, backbone, model_name):
        # create arcface loss that only takes in y_pred and y_true
        arcface = partial(_arcface, m=self.config.model.hyperparameters.m, s=self.config.model.hyperparameters.s)
        arcface.__name__ = "arcface_loss"
        full_model = self.define_full(backbone=backbone, model_name=model_name)
        opt = Adam(lr=self.config.model.hyperparameters.lr, beta_1=self.config.model.hyperparameters.beta1,
                   beta_2=self.config.model.hyperparameters.beta2,
                   clipvalue=self.config.model.hyperparameters.clipvalue,
                   clipnorm=self.config.model.hyperparameters.clipnorm)
        full_model.compile(optimizer=opt, loss=arcface, metrics=["accuracy"])
        return full_model
