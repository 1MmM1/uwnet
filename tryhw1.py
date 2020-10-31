from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
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

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# Convnet: 
#       training accuracy: 0.6904799938201904 = 69%
#       test accuracy: 0.6486999988555908 = 64%
# Fully connected: 
#       training accuracy: 0.563979983329773 = 56%
#       test accuracy: 0.5110999941825867 = 51%
# The fully connected network's test accuracy is worse than the convnet test accuracy when they use a similar number 
# of operations by 13%. I suspect this is  because of how fully connected networks take into account all the information
# from the input, whereas there is more of a spatial information focus in convnets. Put simply, convnets do better because
# they make the assumption that nearby pixels are related and far away pixels are less related. By nature, pixels in the
# top right corner of an image are usually more related to the pixels around it than the pixels in the bottom left corner.
# Fully connected layers try to take in all the information and so they might be confused because they are taking in the
# information from all the pixels in the image instead of just the spatially relevent ones. Convolutional networks are able
# to only focus on the more relevent ones and thus it makese sense that it's more accurate. Because of what we had talked
# about in lecture, I was expecting that the convnet would perform better. Being not only better, but 13% better only further
# confirms this. For more information on how I determinend my arcitecture for the fully connected layers, see below:

# To create the fully connected network, I first calculated how many operations happened in one forward pass of the convnet, 
# which amounted to 1108040, which is about 1.1 million. The breakdown of my calculations are below:
#   First conv layer: (32x32)(3x3x3)(8) = 221184 ops
#   Second conv layer: (16x16)(3x3x8)(16) = 294912 ops
#   Third conv layer: (8x8)(3x3x16)(32) = 294912 ops
#   Fourth conv layer: (4x4)(3x3x32)(64) = 294912 ops
#   Fully connected layer: (1)(256)(10) = 2560 ops
#   Total = 221184 + 294912 + 294912 + 294912 + 2560 = 1108040
# To build my fully connected layer, I knew that I had an input of 3072 (because our images are 32x32x3). I also knew that 
# I should have a total of 5 layers and that my final number of operations had to be about 1108040. I decided to make 
# each hidden layer take the same amount of operations (in lecture it was mentioned that it was good practice to use a 
# similar amout of operations at each layer). This meant that all the hidden layers would be relatively the same size.
#   First fully connected layer: 3072 --> n
#   Second fully connected layer: n --> n
#   Third fully connected layer: n --> n
#   Fourth fully connected layer: n --> n
#   Last fully connected layer: n --> 10
# To find the value of n, I made a polynomial 3n^2 + 3082n = 1108040 by adding up the number of operations at each layer. I
# then solved for n, which came out to be around 280. This resulted in the following network arcitecture:
#   First fully connected layer: 3072 --> 280
#   Second fully connected layer: 280 --> 280
#   Third fully connected layer: 280 --> 280
#   Fourth fully connected layer: 280 --> 280
#   Last fully connected layer: 280 --> 10


