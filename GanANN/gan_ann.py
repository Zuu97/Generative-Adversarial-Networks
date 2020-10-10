from util import get_data
import numpy as np
from matplotlib import pyplot as plt
from variables import*

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger('tensorflow').disabled = True

class ArtifitialGAN(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest , Ytest = get_data()
        self.Xtrain = Xtrain
        self.Xtest = Xtest

    def generatorModel(self):
        inputs = Input(shape=(generator_size,))
        x = Dense(g1, activation=LeakyReLU(alpha=alpha))(inputs)
        x = BatchNormalization(momentum=momentum)(x)
        x = Dense(g2, activation=LeakyReLU(alpha=alpha))(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Dense(g3, activation=LeakyReLU(alpha=alpha))(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Dense(gout, activation='tanh')(x)

        self.generator = Model(inputs, x)

    def discriminatorModel(self):
        inputs = Input(shape=(gout,))
        x = Dense(d1, activation=LeakyReLU(alpha=alpha))(inputs)
        x = Dense(d2, activation=LeakyReLU(alpha=alpha))(x)
        x = Dense(dout, activation='sigmoid')(x)

        self.discriminator = Model(inputs, x)

        self.discriminator.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(0.0002, 0.5),
                    metrics=['accuracy']
                                   )

    def ganModel(self):
        generator_input = Input(shape=(generator_size,))
        generated_img = self.generator(generator_input)

        self.discriminator.trainable = False

        fake_prediction = self.discriminator(generated_img) # while train need to flip the label
        self.gan = Model(generator_input, fake_prediction)
        self.gan.compile(
                loss='binary_crossentropy',
                optimizer=Adam(0.0002, 0.5),
                metrics=['accuracy']
                        )

    def storeImages(self,epoch,rows=5, cols=5):
        noise = np.random.randn(rows*cols, generator_size)
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

        for epoch in range(1, num_epochs+1):
            idxs = np.random.choice(len(self.Xtrain), batch_size)
            real_imgs = self.Xtrain[idxs]
            noise = np.random.randn(batch_size, generator_size)
            generated_imgs = self.generator.predict(noise)

            #Train discriminator
            Dloss_real, Dacc_real = self.discriminator.train_on_batch(
                                                        real_imgs,
                                                        ones
                                                        )
            Dloss_fake, Dacc_fake = self.discriminator.train_on_batch(
                                                        generated_imgs,
                                                        zeros
                                                        )
            Dloss_epoch = (Dloss_real + Dloss_fake) / 2.0
            Dacc_epoch  = (Dacc_real  + Dacc_fake)  / 2.0

            #Train Generator
            noise = np.random.randn(batch_size, generator_size)
            Gloss_epoch = self.gan.train_on_batch(noise, ones)[0]

            Gloss.append(Gloss_epoch)
            Dloss.append(Gloss_epoch)

            if epoch % verbose == 0:
                print("Epoch: {}, Dloss: {}, Dacc: {}, Gloss: {}".format(epoch, Dloss_epoch, Dacc_epoch, Gloss_epoch))
            if epoch % sample_period == 0:
                self.storeImages(epoch)

if __name__ == "__main__":
    model = ArtifitialGAN()
    model.generatorModel()
    model.discriminatorModel()
    model.ganModel()
    model.trainModel()
