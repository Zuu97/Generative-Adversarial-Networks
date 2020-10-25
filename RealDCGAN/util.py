from variables import *
import numpy as np
import pandas as pd
import re
import pickle
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf

def load_image_data():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                        rescale = rescale,
                                                        rotation_range = rotation_range,
                                                        shear_range = shear_range,
                                                        zoom_range = zoom_range,
                                                        width_shift_range=shift_range,
                                                        height_shift_range=shift_range,
                                                        horizontal_flip = True,
                                                        validation_split= validation_split
                                                                    )

    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = ['Cat'],
                                    subset='training',
                                    shuffle = True)

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = ['Cat'],
                                    subset='validation',
                                    shuffle = True)

    return train_generator, validation_generator

# train_generator, validation_generator = load_image_data()