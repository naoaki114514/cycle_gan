import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torchvision.utils as vutils
import itertools
import sys

from custom_dataset import CustomDataset
from custom_dataset import data_transforms

from model import Generator
from model import Discriminator

os.environ["OMP_NUM_THREADS"] = "1"

args = sys.argv#コマンドライン引数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch = 1
epoch_size = 100
lr_d = 0.000014
lr_g = 0.0002
main_folder = os.path.join("./result", args[1])
#main_folder = "./result/D000014G0002_model_save" #結果などのデータを入れる場所
print(main_folder)

os.makedirs(os.path.join(main_folder, "test_{0:03d}".format(epoch_size)), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test_{0:03d}/generated_images_A".format(epoch_size)), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test_{0:03d}/generated_images_B".format(epoch_size)), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test_{0:03d}/real_images_A".format(epoch_size)), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test_{0:03d}/real_images_B".format(epoch_size)), exist_ok=True)

testdataset = CustomDataset(root="./data/test", transform=data_transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=batch, shuffle=True)

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

netG_A2B_path = os.path.join(main_folder, "netG_A2B.pth")
netG_B2A_path = os.path.join(main_folder, "netG_B2A.pth")
netD_A_path = os.path.join(main_folder, "netD_A.pth")
netD_B_path = os.path.join(main_folder, "netD_B.pth")

netG_A2B.load_state_dict(torch.load(netG_A2B_path))#保存したモデルのパラメータの読み込み
netG_B2A.load_state_dict(torch.load(netG_B2A_path))
netD_A.load_state_dict(torch.load(netD_A_path))
netD_B.load_state_dict(torch.load(netD_B_path))

optimizerG = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()),lr=lr_g,betas=(0.5, 0.999))
optimizerD_A = optim.Adam(netD_A.parameters(), lr = lr_d, betas=(0.5, 0.999))
optimizerD_B = optim.Adam(netD_B.parameters(), lr = lr_d, betas=(0.5, 0.999))

criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()


def main(netG_A2B, netG_B2A, netD_A, netD_B, criterion_GAN, criterion_cycle, optimizerG, optimizerD_A, optimizerD_B, batch):
    with torch.no_grad():
        loss_test_G_A2B = 0
        loss_test_G_B2A = 0
        loss_test_D_A = 0
        loss_test_D_B = 0
        loss_test_cycle = 0
        for i, data_test in enumerate(test_dataloader, 0):
            real_A = data_test[0].to(device)
            real_B = data_test[1].to(device)
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)

            #Discriminator Aの評価
                
            #本物を見分ける
            batch_size = real_A.size()[0]
            label = torch.ones(batch_size).to(device)
            output = netD_A(real_A)
            errD_A_real = criterion_GAN(output, label)
                
            #偽物を見分ける
            label = torch.zeros(batch_size).to(device)
            output = netD_A(fake_A.detach())#勾配がGに伝わらないようにdetach()して止める
            errD_A_fake = criterion_GAN(output, label)
                
            loss_test_D_A += errD_A_real.item() + errD_A_fake.item()

            #Discriminator Bの評価
                
            #本物を見分ける
            label = torch.ones(batch_size).to(device)
            output = netD_B(real_B)
            errD_B_real = criterion_GAN(output, label)
                
            #偽物を見分ける
            label = torch.zeros(batch_size).to(device)
            output = netD_B(fake_B.detach())#勾配がGに伝わらないようにdetach()して止める
            errD_B_fake = criterion_GAN(output, label)
                
            loss_test_D_B += errD_B_real.item() + errD_B_fake.item()

            #Generatorの評価
          
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)
            re_A = netG_B2A(fake_B)
            re_B = netG_A2B(fake_A)

            #GAN Loss
            label = torch.ones(batch_size).to(device)
            output1 = netD_A(fake_A)
            output2 = netD_B(fake_B)

            errG_B2A = criterion_GAN(output1, label)
            errG_A2B = criterion_GAN(output2, label)
                
            loss_test_G_B2A += errG_B2A.item()
            loss_test_G_A2B += errG_A2B.item()

            #cycle Loss
            loss_cycle = criterion_cycle(re_A, real_A) + criterion_cycle(re_B, real_B)

            loss_test_cycle += loss_cycle.item()
            
            joined_real_A = torchvision.utils.make_grid(real_A, nrow=1, padding=3)
            joined_real_B = torchvision.utils.make_grid(real_B, nrow=1, padding=3)

            joined_fake_A = torchvision.utils.make_grid(fake_A, nrow=1, padding=3)
            joined_fake_B = torchvision.utils.make_grid(fake_B, nrow=1, padding=3)

            vutils.save_image(joined_fake_A.detach(), os.path.join(main_folder, "test_{0:03d}/generated_images_A/fake_samples_i_{1:04d}.png".format(epoch_size, i+1)),normalize=True)
            vutils.save_image(joined_fake_B.detach(), os.path.join(main_folder, "test_{0:03d}/generated_images_B/fake_samples_i_{1:04d}.png".format(epoch_size, i+1)),normalize=True)
            vutils.save_image(joined_real_A, os.path.join(main_folder, "test_{0:03d}/real_images_A/real_samples_i_{1:04d}.png".format(epoch_size, i+1)), normalize=True)
            vutils.save_image(joined_real_B, os.path.join(main_folder, "test_{0:03d}/real_images_B/real_samples_i_{1:04d}.png".format(epoch_size, i+1)), normalize=True)

        loss_test_G_A2B = loss_test_G_A2B/len(test_dataloader)
        loss_test_G_B2A = loss_test_G_B2A/len(test_dataloader)
        loss_test_cycle = loss_test_cycle/len(test_dataloader)
        loss_test_D_A = loss_test_D_A/len(test_dataloader)
        loss_test_D_B = loss_test_D_B/len(test_dataloader)

        print("loss_test_G_A2B : {0:.10f} loss_test_G_B2A {1:.10f} loss_test_cycle {2:.10f} loss_test_D_A {3:.10f} loss_test_D_B {4:.10f}".format(loss_test_G_A2B, loss_test_G_B2A, loss_test_cycle, loss_test_D_A, loss_test_D_B))
    
