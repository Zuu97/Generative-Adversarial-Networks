import os
lpha = 0.2
momentum = 0.8
batch_size = 32
num_epochs = 30000
size = 28
verbose = 100
sample_period = 200
#generator
generator_size = 100
g1 = 256
g2 = 512
g3 = 1024
gout = 784
alpha = 0.2
momentum = 0.8

# discriminator
d1 = 512
d2 = 256
dout = 1

train_pickle = 'train_data'
test_pickle = 'test_data'
train_path = 'mnist_train.csv'
test_path = 'mnist_test.csv'
gan_prediction_path = os.path.join(os.getcwd(), 'gan_predictions')