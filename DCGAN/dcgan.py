import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Input, Flatten, BatchNormalization, Dropout
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from util import *
from variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DeepConvolutionalGAN(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest , Ytest = get_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        print("train input shape : {}".format(Xtrain.shape))
        print("test  input shape : {}".format(Xtest.shape))
        print("Num. of classes   : {}".format(len(set(self.Ytrain))))

    #     X, Y= get_data()
    #     self.X = X
    #     self.Y = Y
    #     print("input shape : {}".format(X.shape))

    def distriminator(self):
        mobilenet_functional = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        inputs = Input(shape=input_shape)
        x = inputs
        for layer in mobilenet_functional.layers[1:-1]:
            layer.trainable = False
            x = layer(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(dense_1, activation='relu')(x)
        x = Dense(dense_2, activation='relu')(x)
        x = Dense(dense_3, activation='relu')(x)
        x = Dense(dense_3, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(dense_4, activation='sigmoid')(x)
        model = Model(
                    inputs=inputs,
                    outputs=outputs
                    )
        model.summary()
        self.model = model

    def train(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy'],
        )
        self.history = self.model.fit(
                            self.Xtrain,
                            self.Ytrain,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            validation_data = [self.Xtest, self.Ytest]
                            )

    def load_model(self, model_weights):
        loaded_model.load_weights(model_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer=Adam(learning_rate),
                        metrics=['accuracy'],
                        )
        self.model = loaded_model

    def save_model(self, model_weights, model_architecture):
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def run(self):
        if os.path.exists(discriminator_weights):
            self.load_model(discriminator_weights)
        else:
            self.distriminator()
            self.train()
            self.save_model(discriminator_weights, discriminator_architecture)

if __name__ == "__main__":
    model = DeepConvolutionalGAN()
    model.run()