import os
batch_size = 32
num_epochs = 10
size = 28
new_size = 32
verbose = 10
sample_period = 10
image_size = 224
in_channels = 3
color_mode = 'rgb'
input_size = (image_size, image_size, in_channels)
target_size = (image_size, image_size)
keep_prob = 0.3
learning_rate = 0.0002
shear_range = 0.2
zoom_range = 0.15
rotation_range = 20
shift_range = 0.2
rescale = 1./255
validation_split = 0.2

#generator
gen_1 = 7 * 7 * 256
latent_dim = 100
stride1 = (1,1)
stride2 = (2,2)
kernal_size = (5,5)
upconv1_dim = (7, 7, 256)
fs1 = 256
fs2 = 128
fs3 = 64
alpha = 0.2
momentum = 0.8

#discriminator 
dense_1 = 512
dense_2 = 256
dense_3 = 64
dense_4 = 1
model_weights = 'weights/real_gan_weights.h5'
model_architecture = 'weights/real_gan_architecture.json'

train_dir = 'data/'
gan_prediction_path = os.path.join(os.getcwd(), 'real_dcgan_predictions')