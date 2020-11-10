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


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# First, train the conv_net as usual. Then try it with batchnorm. Does it do better??
# Your answer:
#
# In class we learned about annealing your learning rate to get better convergence. We ALSO learned
# that with batch normalization you can use larger learning rates because it's more stable. Increase
# the starting learning rate to .1 and train for multiple rounds with successively smaller learning
# rates. Using just this model, what's the best performance you can get?
# Your answer:

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
# accuracy).

# When using a learning rate of 0.1, we get training accuracy of 0.5148199796676636 (51.4%) and
# test accuracy of 0.5060999989509583 (50.6%).
