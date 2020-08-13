import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def trainer(D_net, G_net, criterion, G_optimizer, D_optimizer, dataloader, num_epochs,channel_noise,device):
    for epoch in range(num_epochs):
        tk0 = tqdm(dataloader, total=int(len(dataloader)))
        for batch_idx,data in enumerate(tk0):
            batch_size = data.shape[0]
            data = data.to(device)
            D_net.zero_grad()
    
            output = D_net(data)
            output = output.reshape(-1)
            label = (torch.ones(output.shape[0])*0.9).to(device)
            lossD_real = criterion(output,label)
            D_x = output.mean().item()

            noise = torch.randn(batch_size,channel_noise,1,1).to(device)
            fake = G_net(noise).to(device)
            label = (torch.ones(batch_size)*0.1).to(device)

            output = D_net(fake.detach()).reshape(-1)
            lossD_fake = criterion(output,label)
    
            lossD = lossD_fake + lossD_real
            lossD.backward()
            D_optimizer.step()

            G_net.zero_grad()
            label = torch.ones(batch_size).to(device)
            output = D_net(fake).reshape(-1)
            lossG = criterion(output, label)
            lossG.backward()
            G_optimizer.step()

            if batch_idx==0:

    
                fig = plt.figure(figsize=(10,7))
                for i , img in enumerate(fake[:5], 1):
                    img = img.cpu().detach().numpy().transpose((1, 2, 0))
                    img = 0.5 * img + 0.5
                    img = np.clip(img, 0, 1)
                    plt.axis('off')
                    plt.subplot(1, 5, i)
                    plt.imshow(img)
                plt.axis('off')
                plt.savefig('Epoch' + str(epoch) + '.png')
                plt.show()
      
