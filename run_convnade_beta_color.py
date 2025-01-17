"""
Trains ConvNADE-Beta-Color"""
import torchvision 
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import numpy as np
import torch
from torch import distributions
import torchvision.datasets as datasets  # Standard datasets
# Transformations we can perform on our dataset for augmentation
import torchvision.transforms as transforms
# Gives easier dataset managment by creating mini batches etc.
from torch.utils.data import (DataLoader,)
from tqdm import tqdm  # For nice progress bar!

from convnade_beta_color import ConvNADEBetaColor

import matplotlib.pyplot as plt

import os

from torchvision.datasets import ImageFolder

import qmcpy as qp

# ------------------------------------------------------------------------------

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 32
image_dim = D * D

ordering_type = 2  # 1 - random permutation. 2 - low-dis ordering.
num_orderings = 1

patch_size = 128
num_patches = image_dim // patch_size

lr = 1e-4  # learning rate 5e-5 for rastor
batch_size = 100 # 100
epochs = 60  # 50
# "MNIST", "FashionMNIST", "OMNIGLOT","FER2013", "CIFAR10", "LHQ".
dataset_name = "LHQ"

start_epoch = 0
checkpoint_save_name = False  # None to not save.
checkpoint_load_name = False  # None to not load.

seeds = 5

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
############### Plotting Generated Samples. ###################################
# ------------------------------------------------------------------------------


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def generate_image_orig_samples(data):
    # Orig. Data.
    show(make_grid(data, nrow=10))
    #torchvision.transforms.ToPILImage()(make_grid(data.view(-1, 1, D, D), nrow=10, padding=2)).show()
    
def generate_image_patches(img, order_type=0):
    gen_patches = []
    
    if (order_type == 0): # Natural
        pixel_ordering = np.arange(image_dim)
    elif (order_type == 1): # Random
        pixel_ordering = np.random.permutation(image_dim)
    else: # LD ordering
        sobol_gen = qp.Sobol(2, randomize=False)# Try No Scrambling. 
        sobol_points = sobol_gen.gen_samples(n=image_dim)
        
        pixel_ordering = (np.floor(sobol_points * D)[:,0] + (D-1 - np.floor(sobol_points * D)[:,1])*D).astype(int)
    
    ds = [128, 1024]
    
    for d in ds:
        mask_d = torch.tensor([float(j in pixel_ordering[:d]) for j in list(range(image_dim))]) 
        mask_3d = torch.cat((mask_d.view(1,D,D), mask_d.view(1,D,D), mask_d.view(1,D,D)), 0)
        x_hat = img.view(3,D,D) * mask_3d.to(device)
    
        gen_patches.append(x_hat)

    data_hat = torch.stack(gen_patches)
    
    torchvision.transforms.ToPILImage()(make_grid(data_hat.view(-1, 3, D, D), nrow=8, padding=2)).show()
    
def generate_gen_compar(nimages, m=0):
    gen_images = []
    select_ordering = model.orderings[0]
    d = m
    
    for n in range(nimages):
        x_hat = torch.rand(3,image_dim).to(device)
        
        mask_d = torch.tensor([float(j in select_ordering[:d]) for j in list(range(image_dim))]).to(device) 
        mask_3d = torch.cat((mask_d.view(1,D,D), mask_d.view(1,D,D), mask_d.view(1,D,D)), 0)
        x_masked = x_hat.view(3,D,D) * mask_3d
        h_0 = torch.cat((x_masked, mask_d.view(1,D,D)), 0)
        h_L = model(h_0).view(6,image_dim) # n_samples x self._input_dim

        h_L_pos = torch.nn.functional.softplus(h_L)
        
        alpha_r = h_L_pos[0, :]
        beta_r = h_L_pos[1, :]
        alpha_g = h_L_pos[2, :]
        beta_g = h_L_pos[3, :]
        alpha_b = h_L_pos[4, :]
        beta_b = h_L_pos[5, :]

        # Sample 'x'.
        if (d < m):
            x_hat[0,select_ordering[d:]] = distributions.Beta(alpha_r[select_ordering[d:(d+patch_size)]], beta_r[select_ordering[d:(d+patch_size)]]).sample()
            x_hat[1,select_ordering[d:]] = distributions.Beta(alpha_g[select_ordering[d:(d+patch_size)]], beta_g[select_ordering[d:(d+patch_size)]]).sample()
            x_hat[2,select_ordering[d:]] = distributions.Beta(alpha_b[select_ordering[d:(d+patch_size)]], beta_b[select_ordering[d:(d+patch_size)]]).sample()
        else:
            x_hat[0,select_ordering[d:]] = alpha_r[select_ordering[d:]] / (alpha_r[select_ordering[d:]] + beta_r[select_ordering[d:]])
            x_hat[1,select_ordering[d:]] = alpha_g[select_ordering[d:]] / (alpha_g[select_ordering[d:]] + beta_g[select_ordering[d:]])
            x_hat[2,select_ordering[d:]] = alpha_b[select_ordering[d:]] / (alpha_b[select_ordering[d:]] + beta_b[select_ordering[d:]])
        
        gen_images.append(x_hat.cpu().view(3,D,D))
    show(make_grid(torch.stack(gen_images).view(-1, 3, D, D), nrow=10))
    #torchvision.transforms.ToPILImage()(make_grid(torch.stack(gen_images).view(-1, 3, D, D), nrow=10, padding=2)).show()

def generate_recon_compar(x):
    gen_images = []
    nimages = len(x)
    select_ordering = model.orderings[0]
    d = patch_size
    
    for n in range(nimages):
        x_hat = torch.zeros(3,image_dim).to(device)
        
        mask_d = torch.tensor([float(j in select_ordering[:d]) for j in list(range(image_dim))]).to(device) 
        mask_3d = torch.cat((mask_d.view(1,D,D), mask_d.view(1,D,D), mask_d.view(1,D,D)), 0)
        x_masked = x[n,:].view(3,D,D) * mask_3d
        mask_3d_ = torch.cat((mask_d.view(1,image_dim), mask_d.view(1,image_dim), mask_d.view(1,image_dim)), 0)
        x_hat = x[n,:].view(3,image_dim) * mask_3d_
        h_0 = torch.cat((x_masked, mask_d.view(1,D,D)), 0)
        h_L = model(h_0).view(6,image_dim) # n_samples x self._input_dim

        h_L_pos = torch.nn.functional.softplus(h_L)
        
        alpha_r = h_L_pos[0, :]
        beta_r = h_L_pos[1, :]
        alpha_g = h_L_pos[2, :]
        beta_g = h_L_pos[3, :]
        alpha_b = h_L_pos[4, :]
        beta_b = h_L_pos[5, :]

        # Sample 'x'.
        x_hat[0,select_ordering[d:]] = alpha_r[select_ordering[d:]] / (alpha_r[select_ordering[d:]] + beta_r[select_ordering[d:]])
        x_hat[1,select_ordering[d:]] = alpha_g[select_ordering[d:]] / (alpha_g[select_ordering[d:]] + beta_g[select_ordering[d:]])
        x_hat[2,select_ordering[d:]] = alpha_b[select_ordering[d:]] / (alpha_b[select_ordering[d:]] + beta_b[select_ordering[d:]])
    
        gen_images.append(x_hat.cpu().view(3,D,D))
    show(make_grid(torch.stack(gen_images).view(-1, 3, D, D), nrow=10))
    #torchvision.transforms.ToPILImage()(make_grid(torch.stack(gen_images).view(-1, 1, D, D), nrow=10, padding=2)).show()

# ------------------------------------------------------------------------------
##################### Save/Load Model ##########################################
# ------------------------------------------------------------------------------


def save_model(epoch, model, optimizer, training_loss, testing_loss, filepath="model/mod1.tar"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': training_loss,
        'test_loss': testing_loss}, filepath)
# ------------------------------------------------------------------------------

def train_run():
    print("Training Model.")
    # enable/disable grad for efficiency of forwarding test batches
    torch.set_grad_enabled(True)
    model.train()

    train_losses = []
    
    pixel_ordering = model.orderings[0]

    d = patch_size

    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        
        data_len = len(data)
        
        data = torch.clamp(data, min=0.00001, max=0.99999)
        
        data_flat = data.view(data_len, 3, -1)

        # We can move these 3 lines out and store them for each patch to save computation.
        mask_d = torch.tensor([float(j in pixel_ordering[:d]) for j in list(range(image_dim))]).to(device)
        mask_3d = torch.cat((mask_d.view(1,D,D), mask_d.view(1,D,D), mask_d.view(1,D,D)), 0)
        
        mask_od = torch.tensor([float(j in pixel_ordering[d:]) 
                                for j in list(range(image_dim))]).to(device)
        
        x_masked = data * mask_3d
        h_0 = torch.cat((x_masked, mask_d.view(1,D,D).repeat(data_len, 1, 1, 1)), 1)

        h_L = model(h_0).view(data_len, 6, -1)
        
        h_L_pos = torch.nn.functional.softplus(h_L)
        
        alpha_r = h_L_pos[:,0, :]
        beta_r = h_L_pos[:,1, :]
        alpha_g = h_L_pos[:,2, :]
        beta_g = h_L_pos[:,3, :]
        alpha_b = h_L_pos[:,4, :]
        beta_b = h_L_pos[:, 5, :]
        
        # Evaluate the loss for red channel.
        loss_r = -torch.matmul(distributions.Beta(alpha_r,
                             beta_r).log_prob(data_flat[:,0,:]), mask_od)
        # average batch values.
        loss_r = torch.mean(1.0 / (image_dim - d) * loss_r)
        
        # Evaluate the loss for green channel.
        loss_g = -torch.matmul(distributions.Beta(alpha_g,
                             beta_g).log_prob(data_flat[:,1,:]), mask_od)
        # average batch values.
        loss_g = torch.mean(1.0 / (image_dim - d) * loss_g)
        
        # Evaluate the loss for blue channel.
        loss_b = -torch.matmul(distributions.Beta(alpha_b,
                             beta_b).log_prob(data_flat[:,2,:]), mask_od)
        # average batch values.
        loss_b = torch.mean(1.0 / (image_dim - d) * loss_b)
        
        loss = (loss_r + loss_g + loss_b) / 3

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient descent or adam step
        optimizer.step()
        
        train_losses.append(loss.data.item())
    print("Train epoch average loss: %f" % (np.mean(train_losses)))

    return np.mean(train_losses)

def test_run(image_loader):
    print("Testing Model")
    # enable/disable grad for efficiency of forwarding test batches
    torch.set_grad_enabled(False)
    model.eval()

    test_losses = []
    
    pixel_ordering = model.orderings[0]

    d = patch_size

    for batch_idx, (data, _) in enumerate(tqdm(image_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        
        data_len = len(data)
        
        data = torch.clamp(data, min=0.00001, max=0.99999)
        
        data_flat = data.view(data_len, 3, -1)

        mask_d = torch.tensor([float(j in pixel_ordering[:d]) for j in list(range(image_dim))]).to(device)
        mask_3d = torch.cat((mask_d.view(1,D,D), mask_d.view(1,D,D), mask_d.view(1,D,D)), 0)
        
        mask_od = torch.tensor([float(j in pixel_ordering[d:]) 
                                for j in list(range(image_dim))]).to(device)
        
        x_masked = data * mask_3d
        h_0 = torch.cat((x_masked, mask_d.view(1,D,D).repeat(data_len, 1, 1, 1)), 1)

        h_L = model(h_0).view(data_len, 6, -1)
        
        h_L_pos = torch.nn.functional.softplus(h_L)
        
        alpha_r = h_L_pos[:,0, :]
        beta_r = h_L_pos[:,1, :]
        alpha_g = h_L_pos[:,2, :]
        beta_g = h_L_pos[:,3, :]
        alpha_b = h_L_pos[:,4, :]
        beta_b = h_L_pos[:,5, :]
        
        # Evaluate the loss for red channel.
        loss_r = -torch.matmul(distributions.Beta(alpha_r,
                             beta_r).log_prob(data_flat[:,0,:]), mask_od)
        # average batch values.
        loss_r = torch.mean(1.0 / (image_dim - d)  * loss_r)
        
        # Evaluate the loss for green channel.
        loss_g = -torch.matmul(distributions.Beta(alpha_g,
                             beta_g).log_prob(data_flat[:,1,:]), mask_od)
        # average batch values.
        loss_g = torch.mean(1.0 / (image_dim - d)  * loss_g)
        
        # Evaluate the loss for blue channel.
        loss_b = -torch.matmul(distributions.Beta(alpha_b,
                             beta_b).log_prob(data_flat[:,2,:]), mask_od)
        # average batch values.
        loss_b = torch.mean(1.0 / (image_dim - d)  * loss_b)
        
        loss = (loss_r + loss_g + loss_b) / 3

        test_losses.append(loss.data.item())

    print("Test epoch average loss: %f" % (np.mean(test_losses)))

    return np.mean(test_losses)

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    
    training_loss = np.empty([seeds, epochs])
    validation_loss = np.empty([seeds, epochs])
    
    testing_loss = []
    
    for seed in range(seeds):
        seed = seed + 1
        #seed = 5
        # Set reproducibility.
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # load the dataset: MNIST, Fashion-MNIST, and Omniglot.
        transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])

        if dataset_name == "MNIST":
            print("loading mnist")

            train_dataset = datasets.MNIST(
                root="dataset/", train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(
                root="dataset/", train=False, transform=transform, download=True)

            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,[50000,10000])

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size, shuffle=True)
        elif dataset_name == "FashionMNIST":
            print("loading fashion mnist")

            train_dataset = datasets.FashionMNIST(
                root="dataset/", train=True, transform=transform, download=True)
            test_dataset = datasets.FashionMNIST(
                root="dataset/", train=False, transform=transform, download=True)

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size, shuffle=True)
        elif dataset_name == "CIFAR10":
            print("loading CIFAR10")

            transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = datasets.CIFAR10(
                root="dataset/", train=True, transform=transform, download=True)
            test_dataset = datasets.CIFAR10(
                root="dataset/", train=False, transform=transform, download=True)
            
            
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [
                                                                        44000, 6000])

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(dataset=valid_dataset,
                                     batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size, shuffle=True)
        elif dataset_name == "FER2013":
            print("loading fer 2013")

            data_dir = os.getcwd() + "\\dataset\\fer_2013\\"

            train_transforms = transforms.Compose([transforms.Resize((D, D)), transforms.ToTensor(),])
            img_transforms = transforms.Compose([transforms.Resize((D, D)), transforms.ToTensor(),])
            train_dataset = ImageFolder(data_dir + 'train', train_transforms)
            test_dataset = ImageFolder(data_dir + 'test', img_transforms)
            
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [
                                                                        24403, 4306])

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(dataset=valid_dataset,
                                     batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size, shuffle=True)

        elif dataset_name == "LHQ":
            print("loading fer LHQ")

            data_dir = os.getcwd() + "\\dataset\\LHQ256\\"

            img_transforms = transforms.Compose([transforms.Resize((D, D)), transforms.ToTensor(),])
            img_dataset = ImageFolder(data_dir, img_transforms)

            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(img_dataset, [
                                                                        63000, 13500, 13500])

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        # construct model and transfer to GPU
        model = ConvNADEBetaColor(D, ordering_type, num_orderings)
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr)
        
        train_loss = []
        valid_loss = []
        
        # #### Load Model ####
        # if checkpoint_load_name:
        #     checkpoint_name = "model/model%d_%d.tar" % (start_epoch,ordering_type)
        #     checkpoint = torch.load(checkpoint_name)
        #     model.load_state_dict(checkpoint['model_state_dict'])
        #     model = model.to(device)
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     training_loss = checkpoint["train_loss"]
        #     testing_loss = checkpoint["test_loss"]

        # start the training
        for epoch in range(start_epoch,epochs,1):
            print("epoch %d" % (epoch, ))

            train_loss.append(train_run())
            valid_loss.append(test_run(valid_loader))
            
            #### Save Model ####
            # if checkpoint_save_name and (epoch+1) % 10 == 0:
            #     checkpoint_name = "model/model%d_%d.tar" % (epoch+1,ordering_type)
            #     save_model(epoch, model, optimizer,
            #                training_loss, testing_loss, filepath=checkpoint_name)    

        print("optimization done. full test set eval:")
        
        training_loss[seed-1, :] = train_loss
        validation_loss[seed-1, :] = valid_loss
        
        loss = test_run(test_loader)
        testing_loss.append(loss)
        
        print(loss)
    
    #### Generate Images ####
    (data, _) = next(iter(test_loader))
    data = data.view(batch_size, 3, D, D).to(device=device)
    data = torch.clamp(data, min=0.0001, max=0.9999)
    
    generate_image_orig_samples(data[:100])
    generate_recon_compar(data[:100])
    
    # train_rand_mean = np.mean(training_loss_Rand, axis=0)
    # train_rand_std = np.std(training_loss_Rand, axis=0) / np.sqrt(seeds)
    
    # plt.plot(list(range(1,epochs+1)), train_rand_mean, label="Random Train Loss", marker='o', linestyle='-', linewidth=1.5) 
    # plt.fill_between(range(epochs),train_rand_mean-2*train_rand_std, train_rand_mean+2*train_rand_std, color='red', alpha=0.2)
    
    # plt.plot(list(range(1,epochs+1)), training_loss_LD, label='LD Train Loss', marker='v', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), training_loss_Rastor, label='Rastor Train Loss', marker='v', linestyle='-', linewidth=1.5)
    
    # plt.xlabel('Epoch'); plt.ylabel('Loss') 
    
    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.legend(fontsize=10, frameon=False)
    # plt.tight_layout()
    # plt.show()
    
    # test_rand_mean = np.mean(testing_loss_Rand, axis=0)
    # test_rand_std = np.std(testing_loss_Rand, axis=0) / np.sqrt(seeds)

    # plt.plot(list(range(1,epochs+1)), test_rand_mean, label="Random Validation Loss", marker='o', linestyle='-', linewidth=1.5) 
    # plt.fill_between(range(epochs),test_rand_mean-2*test_rand_std, test_rand_mean+2*test_rand_std, color='red', alpha=0.2)

    # plt.plot(list(range(1,epochs+1)), testing_loss_LD, label='LD Validation Loss', marker='v', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), testing_loss_Rastor, label='Rastor Validation Loss', marker='v', linestyle='-', linewidth=1.5)

    # plt.xlabel('Epoch'); plt.ylabel('Loss') 

    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.legend(fontsize=10, frameon=False)
    # plt.tight_layout()
    # plt.show()
    
    # # Plot each method
    # plt.plot(list(range(1,epochs+1)), training_loss_Rand, label='Random Train Loss', marker='o', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), training_loss_LD, label='LD Train Loss', marker='v', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), training_loss_Rastor, label='Raster Train Loss', marker='^', linestyle='-', linewidth=1.5)
    
    # # Set axis labels and limits
    # plt.xlabel('Epochs', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    
    # # Add grid, legend, and title
    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.legend(fontsize=10, frameon=False)
    # plt.tight_layout()
    
    # # Show the plot
    # plt.show()
    
    # # Plot each method
    # plt.plot(list(range(1,epochs+1)), testing_loss_Rand, label='Random Test Loss', marker='o', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), testing_loss_LD, label='LD Test Loss', marker='v', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), testing_loss_Rastor, label='Raster Test Loss', marker='^', linestyle='-', linewidth=1.5)
    
    # # Set axis labels and limits
    # plt.xlabel('Epochs', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    
    # # Add grid, legend, and title
    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.legend(fontsize=10, frameon=False)
    # plt.tight_layout()
    
    # # Show the plot
    # plt.show()
