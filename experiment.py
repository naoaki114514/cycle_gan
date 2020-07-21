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

os.makedirs(main_folder, exist_ok=True)
os.makedirs(os.path.join(main_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/generated_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/generated_images_B"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/real_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/real_images_B"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/generated_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/generated_images_B"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/real_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/real_images_B"), exist_ok=True)

save_path_train = os.path.join(main_folder, "loss_train.png")
save_path_test = os.path.join(main_folder, "loss_test.png")

traindataset = CustomDataset(root="./data/train", transform=data_transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=batch, shuffle=True)

testdataset = CustomDataset(root="./data/test", transform=data_transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=batch, shuffle=True)

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

optimizerG = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()),lr=lr_g,betas=(0.5, 0.999))
optimizerD_A = optim.Adam(netD_A.parameters(), lr = lr_d, betas=(0.5, 0.999))
optimizerD_B = optim.Adam(netD_B.parameters(), lr = lr_d, betas=(0.5, 0.999))

criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()

loss_train_G_A2B = []
loss_train_G_B2A = []
loss_train_D_A = []
loss_train_D_B = []
loss_train_cycle = []

loss_test_G_A2B = []
loss_test_G_B2A = []
loss_test_D_A = []
loss_test_D_B = []
loss_test_cycle = []

def main(netG_A2B, netG_B2A, netD_A, netD_B, criterion_GAN, criterion_cycle, optimizerG, optimizerD_A, optimizerD_B, n_epoch, batch):
   
    for epoch in range(n_epoch):
        netG_A2B.train()
        netG_A2B.train()
        netD_A.train()
        netD_B.train()

        loss_train_G_A2B_epoch = 0
        loss_train_G_B2A_epoch = 0
        loss_train_D_A_epoch = 0
        loss_train_D_B_epoch = 0
        loss_train_cycle_epoch = 0
        for i, data_train in enumerate(train_dataloader, 0):
            if data_train[0].size()[0] != batch:
                break
            
            real_A = data_train[0].to(device)
            real_B = data_train[1].to(device)
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)

            #Discriminator Aの学習
            optimizerD_A.zero_grad()

            #本物を見分ける
            batch_size = real_A.size()[0]
            label = torch.ones(batch_size).to(device)
            output = netD_A(real_A)
           
            errD_A_real = criterion_GAN(output, label)
            errD_A_real.backward()

            #偽物を見分ける
            label = torch.zeros(batch_size).to(device)
            output = netD_A(fake_A.detach())#勾配がGに伝わらないようにdetach()して止める
            errD_A_fake = criterion_GAN(output, label)
            errD_A_fake.backward()

            loss_train_D_A_epoch += errD_A_real.item() + errD_A_fake.item()

            optimizerD_A.step()

            #Discriminator Bの学習
            optimizerD_B.zero_grad()

            #本物を見分ける
            label = torch.ones(batch_size).to(device)
            output = netD_B(real_B)
            errD_B_real = criterion_GAN(output, label)
            errD_B_real.backward()

            #偽物を見分ける
            label = torch.zeros(batch_size).to(device)
            output = netD_B(fake_B.detach())#勾配がGに伝わらないようにdetach()して止める
            errD_B_fake = criterion_GAN(output, label)
            errD_B_fake.backward()

            loss_train_D_B_epoch += errD_B_real.item() + errD_B_fake.item()

            optimizerD_B.step()

            #Generatorの学習
            optimizerG.zero_grad()

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
            errG = errG_B2A + errG_A2B

            loss_train_G_B2A_epoch += errG_B2A.item()
            loss_train_G_A2B_epoch += errG_A2B.item()

            #cycle Loss
            loss_cycle = criterion_cycle(re_A, real_A) + criterion_cycle(re_B, real_B)

            loss_train_cycle_epoch += loss_cycle.item()

            errG += loss_cycle
            errG.backward()

            optimizerG.step()

            print("train epoch: [{0:d}/{1:d}][{2:d}/{3:d}] LossD_A: {4:.4f} LossD_B: {5:.4f} LossG_B2A: {6:.4f} LossG_A2B: {7:.4f} Loss_cycle: {8:.4f}".format(epoch+1, n_epoch, i+1, len(train_dataloader), errD_A_real.item() + errD_A_fake.item(), errD_B_real.item() + errD_B_fake.item(), errG_B2A.item(), errG_A2B.item(), loss_cycle.item()))
        

        joined_real_A = torchvision.utils.make_grid(real_A, nrow=2, padding=3)
        joined_real_B = torchvision.utils.make_grid(real_B, nrow=2, padding=3)

        joined_fake_A = torchvision.utils.make_grid(fake_A, nrow=2, padding=3)
        joined_fake_B = torchvision.utils.make_grid(fake_B, nrow=2, padding=3)

        vutils.save_image(joined_fake_A.detach(), os.path.join(main_folder, "train/generated_images_A/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
        vutils.save_image(joined_fake_B.detach(), os.path.join(main_folder, "train/generated_images_B/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
        vutils.save_image(joined_real_A, os.path.join(main_folder, "train/real_images_A/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)
        vutils.save_image(joined_real_B, os.path.join(main_folder, "train/real_images_B/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)

        loss_train_G_A2B.append(loss_train_G_A2B_epoch/len(train_dataloader))
        loss_train_G_B2A.append(loss_train_G_B2A_epoch/len(train_dataloader))
        loss_train_cycle.append(loss_train_cycle_epoch/len(train_dataloader))
        loss_train_D_A.append(loss_train_D_A_epoch/len(train_dataloader))
        loss_train_D_B.append(loss_train_D_B_epoch/len(train_dataloader))

        with torch.no_grad():
            loss_test_G_A2B_epoch = 0
            loss_test_G_B2A_epoch = 0
            loss_test_D_A_epoch = 0
            loss_test_D_B_epoch = 0
            loss_test_cycle_epoch = 0
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
                
                loss_test_D_A_epoch += errD_A_real.item() + errD_A_fake.item()

                #Discriminator Bの評価
                
                #本物を見分ける
                label = torch.ones(batch_size).to(device)
                output = netD_B(real_B)
                errD_B_real = criterion_GAN(output, label)
                
                #偽物を見分ける
                label = torch.zeros(batch_size).to(device)
                output = netD_B(fake_B.detach())#勾配がGに伝わらないようにdetach()して止める
                errD_B_fake = criterion_GAN(output, label)
                
                loss_test_D_B_epoch += errD_B_real.item() + errD_B_fake.item()

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
                
                loss_test_G_B2A_epoch += errG_B2A.item()
                loss_test_G_A2B_epoch += errG_A2B.item()

                #cycle Loss
                loss_cycle = criterion_cycle(re_A, real_A) + criterion_cycle(re_B, real_B)

                loss_test_cycle_epoch += loss_cycle.item()
                print("test epoch: [{0:d}/{1:d}][{2:d}/{3:d}] LossD_A: {4:.4f} LossD_B: {5:.4f} LossG_B2A: {6:.4f} LossG_A2B: {7:.4f} Loss_cycle: {8:.4f}".format(epoch+1, n_epoch, i+1, len(test_dataloader), errD_A_real.item() + errD_A_fake.item(), errD_B_real.item() + errD_B_fake.item(), errG_B2A.item(), errG_A2B.item(), loss_cycle.item()))
            
            joined_real_A = torchvision.utils.make_grid(real_A, nrow=2, padding=3)
            joined_real_B = torchvision.utils.make_grid(real_B, nrow=2, padding=3)

            joined_fake_A = torchvision.utils.make_grid(fake_A, nrow=2, padding=3)
            joined_fake_B = torchvision.utils.make_grid(fake_B, nrow=2, padding=3)

            vutils.save_image(joined_fake_A.detach(), os.path.join(main_folder, "test/generated_images_A/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
            vutils.save_image(joined_fake_B.detach(), os.path.join(main_folder, "test/generated_images_B/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
            vutils.save_image(joined_real_A, os.path.join(main_folder, "test/real_images_A/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)
            vutils.save_image(joined_real_B, os.path.join(main_folder, "test/real_images_B/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)

            loss_test_G_A2B.append(loss_test_G_A2B_epoch/len(test_dataloader))
            loss_test_G_B2A.append(loss_test_G_B2A_epoch/len(test_dataloader))
            loss_test_cycle.append(loss_test_cycle_epoch/len(test_dataloader))
            loss_test_D_A.append(loss_test_D_A_epoch/len(test_dataloader))
            loss_test_D_B.append(loss_test_D_B_epoch/len(test_dataloader))
    
    #モデルを保存
    torch.save(netG_A2B.state_dict(), os.path.join(main_folder, "netG_A2B.pth"))
    torch.save(netG_B2A.state_dict(), os.path.join(main_folder, "netG_B2A.pth"))
    torch.save(netD_A.state_dict(), os.path.join(main_folder, "netD_A.pth"))
    torch.save(netD_B.state_dict(), os.path.join(main_folder, "netD_B.pth"))

def show_loss_train(save_path):
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.ylim([0,3])
    x = [i for i in range(len(loss_train_G_A2B))]
    plt.plot(x, loss_train_G_A2B, color="r", label='G_B')
    plt.plot(x, loss_train_G_B2A, color="g", label='G_A')
    plt.plot(x, loss_train_D_A, color="b", label='D_A')
    plt.plot(x, loss_train_D_B, color="c", label='D_B')
    plt.plot(x, loss_train_cycle, color="y", label='cycle')
    fig.legend()
    fig.savefig(save_path)


def show_loss_test(save_path):
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.ylim([0,3])
    x = [i for i in range(len(loss_test_G_A2B))]
    plt.plot(x, loss_test_G_A2B, color="r", label='G_B')
    plt.plot(x, loss_test_G_B2A, color="g",label='G_A')
    plt.plot(x, loss_test_D_A, color="b", label='D_A')
    plt.plot(x, loss_test_D_B, color="c",label='D_B')
    plt.plot(x, loss_test_cycle, color="y", label='cycle')
    fig.legend()
    fig.savefig(save_path)



if __name__ == '__main__':
    main(netG_A2B=netG_A2B, netG_B2A=netG_B2A, netD_A=netD_A, netD_B=netD_B, criterion_GAN=criterion_GAN, criterion_cycle=criterion_cycle, optimizerG=optimizerG, optimizerD_A=optimizerD_A, optimizerD_B=optimizerD_B, n_epoch=epoch_size, batch=batch)
    show_loss_train(save_path_train)
    show_loss_test(save_path_test)
    print("finish")
