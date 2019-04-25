import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.utils import save_image
from timeit import default_timer as timer
import datetime
from PIL import Image
from data0 import PairedDataset
import glob
import os

# ============= parameters ======================
num_epochs = 200
batch_size = 64
learning_rate = .0001
model_name = 'autoencoder_split_1_image_200_epochs_64batch'
folder_name = './'+model_name
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
# ===============================================

# ============================== Begin Log =======================
now = datetime.datetime.now()
with open(folder_name+'/log_'+str(now)+'.txt', 'a') as log:
    log.write('Epochs {},batch size {},learning rate:{:.6f}\n'.format(num_epochs, batch_size, learning_rate))
# ================================================================

# ==================== Load Data ===================================
transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
# The parent directories of the images (combined and two splits)
parent_A = './SplitData_MNIST/train/combined/'
parent_B = './SplitData_MNIST/train/splitA/'
parent_C = './SplitData_MNIST/train/splitB/'

# create the triplet structured data set
dset = PairedDataset(parent_A, parent_B, parent_C, transform=transform)
data_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
# ==================================================================
# ================== Test If A GPU is available ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if str(device) == "cuda":
    print('Using GPU')
else:
    print('Using CPU')

# ====================================================================

# ====================== Autoencoder =================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.ReLU(),
            nn.Linear(28*28,256),
            nn.ReLU()
            )

        self.decoder = nn.Sequential(
            nn.Linear(256, 28*28),
            nn.ReLU(),
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid()
            )


    def forward(self, x):
        x = x.view(-1, 28*28)
        out = self.encoder(x)
        out = self.decoder(out)
        return out

# ======================================================================

# ==================== Set to device (CPU or GPU) ======================
net = Autoencoder().to(device)
content_criterion = nn.BCELoss().to(device)
optim_net = torch.optim.Adam(net.parameters(), lr=learning_rate)

# ======================================================================

# ================== Extract training sample ===========================
test_sample = next(iter(data_loader))
sample_combined, sample_A, sample_B = test_sample
sample_combined = Variable(sample_combined).to(device)
sample_A = Variable(sample_A).to(device)
sample_B = Variable(sample_B).to(device)
if not os.path.exists(folder_name+'/split_images'):
    os.mkdir(folder_name+'/split_images')
save_image(sample_combined.cpu().data, folder_name+'/split_images/combined.png')
save_image(sample_A.cpu().data, folder_name+'/split_images/sample_A.png')
save_image(sample_B.cpu().data, folder_name+'/split_images/sample_B.png')
# ======================================================================

# ================= Train Autoencoder ==================================
AE_loss = list()
for epoch in range(num_epochs):
    for i, (comb,A,B) in enumerate(data_loader):
        Input = Variable(comb).to(device)
        Target = Variable(A.view(-1,28*28*1)).to(device)
        output = net(Input)
        optim_net.zero_grad()
        autoencoder_loss = content_criterion(output, Target)
        autoencoder_loss.backward()
        optim_net.step()

        if i % 50 == 0:
            with open(folder_name+'/log_'+str(now)+'.txt', 'a') as log:
                log.write('Epoch [{}/{}], Step {}, Autoencoder loss:{:.4f}\n'.format(epoch, num_epochs, i, autoencoder_loss.item()))
            AE_loss.append(autoencoder_loss.item())
    if epoch % 1 == 0:
        output_sample = net(sample_combined)
        save_image(output_sample.view(-1,1,28,28).cpu().data, folder_name+'/split_images/image_{}A.png'.format(epoch))
    if epoch % 10 ==0:
        if not os.path.exists(folder_name+'/saved_models'):
            os.mkdir(folder_name+'/saved_models')
        epoch_save_autoencoder = folder_name+'/saved_models/autoencoder_Epoch_'+str(epoch)+'.pth'
        torch.save(net.state_dict(), epoch_save_autoencoder)
