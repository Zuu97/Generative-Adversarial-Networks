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
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy'],
        )

        self.distriminator_model = model

    def generator(self):
        model = Sequential()
        n_nodes = c1 * h1 * w1
        inputs = Input(shape=(latent_dim,))
        x = Dense(n_nodes)(inputs)  
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((h1, w1, c1))(x)   

        x = Conv2DTranspose(c2, (f1,f1), strides=(s1,s1), padding='same')(x) 
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(c3, (f2,f2), strides=(s2,s2), padding='same')(x)   
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        outputs = Conv2D(c4, (f3,f3), activation='sigmoid', padding='same')(x)
        
        model = Model(
                    inputs=inputs,
                    outputs=outputs
                    )
                                                
        model.summary()

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

    def storeImages(self,epoch,rows=5, cols=5):
        noise = np.random.randn(rows*cols, latent_dim)
        generated_imgs = self.generator.predict(noise)

        generated_imgs = (generated_imgs + 1) * 0.5

        fig, axes = plt.subplots(rows, cols)
        idx = 0
        for i in range(rows):
            for j in range(rows):
                axes[i,j].imshow(generated_imgs[idx].reshape(size, size), cmap='gray')
                axes[i,j].axis('off')
                idx += 1
        fig_name = os.path.join(gan_prediction_path, 'gan'+str(epoch)+'.png')
        fig.savefig(fig_name)
        plt.close()

    def trainModel(self):
        ones = np.ones(batch_size)
        zeros = np.zeros(batch_size)

        Gloss = []
        Dloss = []

        n_batches = len(self.Xtrain) // num_epochs
        for epoch in range(1, num_epochs+1):
            for i in n_batches:
                idxs = np.random.choice(len(self.Xtrain), batch_size)
                noise = np.random.randn(batch_size, latent_dim)

                real_imgs = self.Xtrain[i*batch_size : (i+1)*batch_size]
                fake_imgs = self.generator.predict(noise)

                #Train discriminator
                Dloss_real, Dacc_real = self.discriminator_model.train_on_batch(
                                                                    real_imgs,
                                                                    ones
                                                                    )
                Dloss_fake, Dacc_fake = self.discriminator_model.train_on_batch(
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

                if epoch % verbose == 0:
                    print("Epoch: {}, Dloss: {}, Dacc: {}, Gloss: {}".format(epoch, Dloss_epoch, Dacc_epoch, Gloss_epoch))
                if epoch % sample_period == 0:
                    self.storeImages(epoch)

    @staticmethod
    def load_model(disc_loss):
        json_file = open(model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model.load_weights(model_weights)
        return model

    @staticmethod
    def save_model(model):
        model_json = model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_weights)

if __name__ == "__main__":
    model = DeepConvolutionalGAN()
    model.distriminator()
    model.generator_model()