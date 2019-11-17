import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, Activation, Lambda, \
    GlobalAveragePooling2D

from base.base_model import BaseModel
from utils.layers import L2Normalize, Arccos


class Res50(BaseModel):
    # Backbone is used for valid and prediction and only outptus 512-sized feature embedding
    def define_backbone(self,model_name: str):
        input = Input(shape=(self.config.data.img_size, self.config.data.img_size, self.config.data.img_channels),
                      name="img_in")  # check and delete if this is not neccessary
        res50 = tf.keras.applications.ResNet50(include_top=False, input_shape=(None, None, 3))
        x = res50(input)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        # x = BatchNormalization()(res50.output)
        x = Dropout(0.3)(x)
        x = Dense(512)(x)
        feature = BatchNormalization()(x)
        backbone = Model(inputs=input, outputs=feature, name=model_name)
        return backbone

    def build_backbone(self, model_name):
        return

    # full model is the one with softmax at the end for meta learning stage. It is made with the backbone
    def define_full(self, backbone,model_name):
        x = Lambda(lambda _x: tf.math.l2_normalize(_x))(backbone.output)
        x = L2Normalize(Dense(self.config.data.n_classes))(x)
        x = Arccos(self.config.model.hyperparameters.m, self.config.model.hyperparameters.s)(x)
        x = Activation("softmax")(x)
        full_model = Model(inputs=backbone.inputs, outputs=x, name=model_name)
        return full_model

    def build_full(self, model_name):
        return
