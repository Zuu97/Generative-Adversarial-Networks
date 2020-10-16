import os
batch_size = 256
num_epochs = 20
size = 28
verbose = 100
sample_period = 200
input_shape = (32, 32, 3)
in_channels = 3
keep_prob = 0.3
learning_rate = 0.0002
padding = ((0, 0), (2,2), (2,2), (0, 0))

#generator
latent_dim = 100
c1, h1, w1 = 256, 8, 8
c2, f1, s1 = 128, 4, 2
c3, f2, s2 = 64, 4, 2
c4, f3 = 3, 7
alpha = 0.2
momentum = 0.8

#discriminator 
dense_1 = 512
dense_2 = 256
dense_3 = 64
dense_4 = 1
model_weights = 'weights/gan_weights.h5'
model_architecture = 'weights/gan_architecture.json'

train_path = 'data/mnist_train.csv'
test_path = 'data/mnist_test.csv'
gan_prediction_path = os.path.join(os.getcwd(), 'gan_predictions')