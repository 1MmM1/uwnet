from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def batched_conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .2
momentum = .9
decay = .005

m = conv_net()
# m = batched_conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How
# does it affect convergence? How does it affect what magnitude of learning rate you can use? Write
# down any observations from your experiments:
# When trained as with no batch normalization layers, we get a training accuracy of
# 0.397599995136261 (39.8%) and test accuracy of 0.4000999927520752 (40%).
# When trained with a batch normalization layer immediately after each convolution, we
# get a training accuracy of 0.559719979763031 (56%) and a test accuracy of
# 0.5454999804496765 (54.5%)
# We can see just by comparing the training accuracies and test accuracies that adding
# batch normalization makes our model do better (i.e. achieve a better train and test
# accuracy) when using the same learning rate (0.01 in this case) and hyperparameters.

# Adding batch normalization also allows us to use higher learning rates without sacrificing a lot of 
# accuracy. When using a learning rate of 0.1, we get training accuracy of 0.5148199796676636 (51.4%) and
# test accuracy of 0.5060999989509583 (50.6%). When we use the same learning rate with a regular convolutional
# neural network, it only achieved an accuracy of ~39% on both test and training sets.
# To achieve the same level of accuracy using the batch normalization that we got using the normal convolutional
# network, I had to use increase the learning rate twentyfold from 0.01 to 0.2.
# I also noticed that using batch normalization also made the model converge faster (loss decreased in
# about 50 iterations compared to 100 iterations).
