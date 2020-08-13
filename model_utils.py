import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self,channels_img,features_d):
    super(Discriminator,self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(in_channels = channels_img, out_channels = features_d, kernel_size= 4, stride = 2, padding = 1),
        nn.LeakyReLU(0.2),
        
        nn.Conv2d(in_channels = features_d, out_channels = features_d*2, kernel_size= 4, stride = 2, padding = 1),
        nn.BatchNorm2d(num_features = features_d*2 ),
        nn.LeakyReLU(0.2),
        
        nn.Conv2d(in_channels = features_d*2, out_channels = features_d*4, kernel_size= 4, stride = 2, padding = 1),
        nn.BatchNorm2d(num_features = features_d*4),
        nn.LeakyReLU(0.2),

        nn.Conv2d(in_channels = features_d*4, out_channels = features_d*8, kernel_size= 4, stride = 2, padding = 1),
        nn.BatchNorm2d(num_features = features_d*8),
        nn.LeakyReLU(0.2),

        nn.Conv2d(in_channels = features_d*8, out_channels = 1, kernel_size = 4, stride = 2, padding = 0),
        nn.Sigmoid()
      )
    
  def forward(self,x):
      return self.model(x)

class Generator(nn.Module):
  def __init__(self,noise,channels_img, features_g):
    super(Generator,self).__init__()
    self.model = nn.Sequential(
        nn.ConvTranspose2d(in_channels = noise, out_channels = features_g*16, kernel_size = 4, stride = 1, padding = 0),
        nn.BatchNorm2d(num_features = features_g*16),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels = features_g*16, out_channels = features_g*8, kernel_size = 4, stride = 2, padding = 1),
        nn.BatchNorm2d(num_features = features_g*8),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels = features_g*8, out_channels = features_g*4, kernel_size = 4, stride = 2, padding = 1),
        nn.BatchNorm2d(num_features = features_g*4),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels = features_g*4, out_channels = features_g*2, kernel_size = 4, stride = 2, padding = 1),
        nn.BatchNorm2d(num_features = features_g*2),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels = features_g*2, out_channels = channels_img, kernel_size = 4, stride = 2, padding = 1),
        nn.Tanh()
    )
  
  def forward(self,x):
    return self.model(x)

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02) 
            m.bias.data.fill_(0)