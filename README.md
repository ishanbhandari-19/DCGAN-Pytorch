# DCGAN-with-Pytorch
Implentation of [DCGAN](https://arxiv.org/abs/1511.06434) on Pytorch framework and trained on The Stanford Dogs Dataset.
## DCGAN
<img src='Generated_images/DCGAN.png' />
In reality, it is very hard to train GANs as its training is very unstable and requires a careful selection of hyperparameters.

**DCGANS** try to solve some of the problems of GANs by using ConvNets in the Generator and Discriminator. The paper also proposes some architectual contraints on the ConvNets that helps stabilize GAN training. Some of these constraints are:

1) The generator uses **tanh** activation in the final layer while the discriminator uses **leaky-relu** activation in all the layers.

2) Using only convolutional layers in the Generator and the Discriminator by increasing the stride.

3) Using BatchNormalization in both the Generator and the Discriminator during training.

## Dataset
[The Stanford Dogs Dataset](https://www.kaggle.com/c/generative-dog-images/data)

