import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
import numpy as np
from tensorflow.keras.models import model_from_json, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Input, Flatten, BatchNormalization, Dropout, LeakyReLU, Reshape, Conv2DTranspose
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from util import *
from variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
 
        python ignore -w dcgan.py

'''
class RealDCGAN(object):
    def __init__(self):
        train_generator, validation_generator = load_image_data()
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.train_step = self.train_generator.samples // batch_size
        self.validation_step = self.validation_generator.samples // batch_size

    def distriminator(self):
        mobilenet_functional = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_size)
        inputs = Input(shape=input_size)
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
                    outputs=outputs,
                    name='Discriminator'
                    )
        # model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy'],
        )

        self.distriminator_model = model

    def generator(self):
        inputs = Input(shape=(latent_dim,))
        x = Dense(gen_1, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape(upconv1_dim)(x) # (7, 7, 256)

        x = Conv2DTranspose(fs1, kernal_size, strides=stride2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x) # (14, 14, 256)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(fs2, kernal_size, strides=stride2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x) # (28, 28, 128)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(fs3, kernal_size, strides=stride2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x) # (56, 56, 64)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(fs3, kernal_size, strides=stride2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x) # (112, 112, 64)
        x = LeakyReLU()(x)

        outputs = Conv2DTranspose(in_channels, kernal_size, strides=stride2, padding='same', use_bias=False, activation='tanh')(x)
                                    # (224, 224, 3)
        
        model = Model(
                    inputs=inputs,
                    outputs=outputs,
                    name='Generator'
                    )
                                                
        # model.summary()

        self.generator_model = model

    def ganModel(self):
        generator_input = Input(shape=(latent_dim,))
        generated_img = self.generator_model(generator_input)

        self.distriminator_model.trainable = False

        fake_prediction = self.distriminator_model(generated_img) # while train need to flip the label
        self.gan = Model(generator_input, fake_prediction)
        self.gan.compile(
                loss='binary_crossentropy',
                optimizer=Adam(learning_rate),
                metrics=['accuracy']
                        )
        # self.gan.summary()

    def storeImages(self,epoch,Iteration,rows=2, cols=2):
        noise = np.random.randn(rows*cols, latent_dim)
        fake_imgs = self.generator_model.predict(noise)   # 25, 32, 32, 3

        fake_imgs = (fake_imgs + 1) * 0.5 # rescale between 0 and 1

        fig, axes = plt.subplots(rows, cols)
        idx = 0
        for i in range(rows):
            for j in range(rows):
                axes[i,j].imshow(fake_imgs[idx].reshape(input_size), cmap='gray')
                axes[i,j].axis('off')
                idx += 1
        fig_name = os.path.join(gan_prediction_path, 'dcgan'+str(epoch)+'_'+str(Iteration)+'.png')
        fig.savefig(fig_name)
        plt.close()

    def trainModel(self):
        ones = np.ones(batch_size)
        zeros = np.zeros(batch_size)

        Gloss = []
        Dloss = []

        for epoch in range(1, num_epochs+1):
            Iteration = 1
            for real_imgs, real_labels in self.train_generator:
                # idxs = np.random.choice(len(self.Xtrain), batch_size)
                noise = np.random.randn(batch_size, latent_dim)
                fake_imgs = self.generator_model.predict(noise)

                #Train discriminator
                Dloss_real, Dacc_real = self.distriminator_model.train_on_batch(
                                                                    real_imgs,
                                                                    ones
                                                                    )
                Dloss_fake, Dacc_fake = self.distriminator_model.train_on_batch(
                                                                    fake_imgs,
                                                                    zeros
                                                                    )
                Dloss_epoch = (Dloss_real + Dloss_fake) / 2.0
                Dacc_epoch  = (Dacc_real  + Dacc_fake)  / 2.0

                #Train Generator
                noise = np.random.randn(batch_size, latent_dim)
                Gloss_epoch = self.gan.train_on_batch(noise, ones)[0]

                Gloss.append(Gloss_epoch)
                Dloss.append(Gloss_epoch)

                if Iteration % verbose == 0:
                    print("Epoch: {}, Iteration: {}, Dloss: {}, Dacc: {}, Gloss: {}".format(epoch, Iteration, Dloss_epoch, Dacc_epoch, Gloss_epoch))
                if Iteration % sample_period == 0:
                    self.storeImages(epoch, Iteration)
                Iteration += 1
                print(Iteration)

    def load_model(self):
        json_file = open(model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.gan = model_from_json(loaded_model_json)
        self.gan.load_weights(model_weights)

    def save_model(self):
        model_json = self.gan.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.gan.save_weights(model_weights)

    def run(self):
        self.distriminator()
        self.generator()
        self.ganModel()

        if os.path.exists(model_architecture):
            self.load_model()
        else:
            self.trainModel()
            self.save_model()


if __name__ == "__main__":
    model = RealDCGAN()
    model.run()