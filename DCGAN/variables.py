import os
batch_size = 256
num_epochs = 20
size = 28
verbose = 100
sample_period = 200
input_shape = (32, 32, 3)
in_channels = 3
keep_prob = 0.3
learning_rate = 0.0001
padding = ((0, 0), (2,2), (2,2), (0, 0))

#generator
generator_size = 100
g1 = 256
g2 = 512
g3 = 1024
gout = 784
alpha = 0.2
momentum = 0.8

#discriminator 
dense_1 = 512
dense_2 = 256
dense_3 = 64
dense_4 = 1
discriminator_weights = 'discriminator_weights.h5'
discriminator_architecture = 'discriminator_architecture.json'

train_path = 'mnist_train.csv'
test_path = 'mnist_test.csv'
gan_prediction_path = os.path.join(os.getcwd(), 'gan_predictions')