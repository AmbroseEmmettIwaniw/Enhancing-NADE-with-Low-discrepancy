"""
Trains ConvNADE model on Binarized MNIST.
"""
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

from convnade_bernoulli import ConvNADEBernoulli

import matplotlib.pyplot as plt

import qmcpy as qp

# ------------------------------------------------------------------------------

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 32
image_dim = D * D

ordering_type = 1  # 1 - random permutation. 2 - low-dis ordering.
num_orderings = 1 # Number of orderings. Not used.

patch_size = 128

lr = 1e-4  # learning rate.
batch_size = 100  # 100
epochs = 60  # 50
# "MNIST", "FashionMNIST", "OMNIGLOT".
dataset_name = "MNIST"

start_epoch = 0
checkpoint_save_name = True  # None to not save.
checkpoint_load_name = False  # None to not load.

seeds = 5 # number of seeds

# Define the stochastic binarization transform
class StochasticBinarization(object):
    def __call__(self, img):
        img = np.array(img, dtype=np.float32) # EDIT: MNIST is already [0.1].
        binarized_img = np.random.binomial(1, img).astype(np.float32)  # Binarized
        return torch.tensor(binarized_img)

class RoundBinary(object):
    def __call__(self, img):
        return torch.round(img)

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

def generate_image_recon_samples(data, model):
    # Model generated samples.
    data_hat = model.recon_sample(data)
    show(make_grid(data_hat.view(-1, 1, D, D), nrow=10))
    #torchvision.transforms.ToPILImage()(make_grid(data_hat.view(-1, 1, D, D), nrow=10, padding=2)).show()


def generate_image_recon_part_samples(data, model, patch_len=200):
    # Model generated samples.
    data_hat = model.recon_part_sample(data, patch_len)
    show(make_grid(data_hat.view(-1, 1, D, D), nrow=10))
    #torchvision.transforms.ToPILImage()(make_grid(data_hat.view(-1, 1, D, D), nrow=10, padding=2)).show()


def generate_image_gen_samples(model, nsamples):
    # Model generated samples.
    data_hat = model.gen_sample(nsamples)
    show(make_grid(data_hat.view(-1, 1, D, D), nrow=10))
    #torchvision.transforms.ToPILImage()(make_grid(data_hat.view(-1, 1, D, D), nrow=10, padding=2)).show()
    
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
    
    ds = [128*(i+1) for i in range(8)]
    
    for d in ds:
        mask_d = torch.tensor([float(j in pixel_ordering[:d]) for j in list(range(image_dim))]) 
        x_hat = img.view(image_dim) * mask_d.to(device)
    
        gen_patches.append(x_hat)

    data_hat = torch.stack(gen_patches)
    
    torchvision.transforms.ToPILImage()(make_grid(data_hat.view(-1, 1, D, D), nrow=8, padding=2)).show()
    
def generate_gen_compar(nimages, m=0):
    gen_images = []
    select_ordering = model.orderings[0]
    d = patch_size
    
    for n in range(nimages):
        x_hat = torch.zeros(image_dim).to(device)
        
        mask_d = torch.tensor([float(j in select_ordering[:d]) for j in list(range(image_dim))]).to(device) 
        x_masked =  x_hat * mask_d
        h_0 = torch.cat((x_masked.view(1,D,D), mask_d.view(1,D,D)), 0)
        h_L = model(h_0).view(image_dim) # n_samples x self._input_dim
        
        h_L = torch.nn.functional.sigmoid(h_L)
        
        # Sample 'x'.
        if (d < m):
            x_hat[select_ordering[d:]] = distributions.Bernoulli(h_L[select_ordering[d:]]).sample()
        else:
            x_hat[select_ordering[d:]] = torch.round(h_L[select_ordering[d:]])
            
        
        gen_images.append(x_hat.cpu().view(1,D,D))
    show(make_grid(torch.stack(gen_images).view(-1, 1, D, D), nrow=10))
    #torchvision.transforms.ToPILImage()(make_grid(torch.stack(gen_images).view(-1, 1, D, D), nrow=10, padding=2)).show()

def generate_recon_compar(x):
    gen_images = []
    nimages = len(x)
    select_ordering = model.orderings[0]
    d = patch_size
    
    for n in range(nimages):
        x_hat = torch.zeros(image_dim).to(device)
        
        mask_d = torch.tensor([float(j in select_ordering[:d]) for j in list(range(image_dim))]).to(device) 
        x_masked =  x[n,:] * mask_d.view(1,D,D)
        #gen_images.append(x_masked.cpu())
        h_0 = torch.cat((x_masked.view(1,D,D), mask_d.view(1,D,D)), 0)
        
        h_L = model(h_0).view(image_dim) # n_samples x self._input_dim
        
        h_L = torch.nn.functional.sigmoid(h_L)

        # Sample 'x'.
        x_hat[select_ordering[d:]] = torch.round(h_L[select_ordering[d:]])
            
        gen_images.append(x_hat.cpu().view(1,D,D))
    show(make_grid(torch.stack(gen_images).view(-1, 1, D, D), nrow=10))
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

def NLL(test_loader):
    print("NLL Model")
    # enable/disable grad for efficiency of forwarding test batches
    torch.set_grad_enabled(False)
    model.eval()

    test_losses = []

    test_len = 0.0
    
    pixel_ordering = model.orderings[0]

    for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
        data_len = len(data)
        test_len += data_len
        # Get data to cuda if possible
        data = data.view(data_len, -1).to(device=device)
        
        data_flat = data

        data = data.view(data_len, 1, D, D)

        d = patch_size
        
        mask_d = torch.tensor([float(j in pixel_ordering[:d])
                              for j in list(range(image_dim))]).to(device)
        mask_od = torch.tensor([float(j in pixel_ordering[d:]) for j in list(range(image_dim))]).to(device)
        
        x_masked = data * mask_d.view(1,D,D)
        h_0 = torch.cat((x_masked, mask_d.view(1,D,D).repeat(data_len, 1, 1, 1)), 1)

        h_L = model(h_0).view(data_len, -1)

        prob = torch.nn.functional.sigmoid(h_L)

        # Evaluate the binary cross entropy loss.
        loss = -torch.matmul(distributions.Bernoulli(prob).log_prob(data_flat), mask_od).sum()
        # loss = torch.mean(loss) # average batch values.

        test_losses.append(loss.data.item())

    print("NLL epoch average loss: %f" % (np.sum(test_losses) / test_len))

    return np.sum(test_losses) / test_len

def train_run():
    print("Training Model.")
    # enable/disable grad for efficiency of forwarding test batches
    torch.set_grad_enabled(True)
    model.train()

    train_losses = []

    d = patch_size

    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data_len = len(data)
        # Get data to cuda if possible
        data = data.view(data_len, -1).to(device=device)
        
        pixel_ordering = model.orderings[0]
        data_flat = data

        data = data.view(data_len, 1, D, D)

        mask_d = torch.tensor([float(j in pixel_ordering[:d])
                              for j in list(range(image_dim))]).to(device)
        #mask_od = (mask_d + 1) % 2  # m_o>=d. 1024x1
        mask_od = torch.tensor([float(j in pixel_ordering[d:]) 
                                for j in list(range(image_dim))]).to(device)
        
        x_masked = data * mask_d.view(1,D,D)
        h_0 = torch.cat((x_masked, mask_d.view(1,D,D).repeat(data_len, 1, 1, 1)), 1)

        h_L = model(h_0).view(data_len, -1)

        # Evaluate the loss.
        loss = torch.matmul(torch.nn.functional.binary_cross_entropy_with_logits(h_L,data_flat, reduction='none'), mask_od)
        # average batch values.
        loss = torch.mean(1.0 / (image_dim - patch_size) * loss)
        train_losses.append(loss.data.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient descent or adam step
        optimizer.step()
        
    print("Train epoch average loss: %f" % (np.mean(train_losses)))

    return np.mean(train_losses)

def test_run(test_loader):
    print("Testing Model")
    # enable/disable grad for efficiency of forwarding test batches
    torch.set_grad_enabled(False)
    model.eval()

    test_losses = []

    d = patch_size

    for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
        data_len = len(data)
        # Get data to cuda if possible
        data = data.view(data_len, -1).to(device=device)

        pixel_ordering = model.orderings[0]
        
        data_flat = data

        data = data.view(data_len, 1, D, D)

        mask_d = torch.tensor([float(j in pixel_ordering[:d])
                              for j in list(range(image_dim))]).to(device)
        #mask_od = (mask_d + 1) % 2  # m_o>=d. 1024x1
        mask_od = torch.tensor([float(j in pixel_ordering[d:]) 
                                for j in list(range(image_dim))]).to(device)

        x_masked = data * mask_d.view(1,D,D)
        h_0 = torch.cat((x_masked, mask_d.view(1,D,D).repeat(data_len, 1, 1, 1)), 1)

        h_L = model(h_0).view(data_len, -1)

        # Evaluate the loss.
        loss = torch.matmul(torch.nn.functional.binary_cross_entropy_with_logits(h_L,data_flat, reduction='none'), mask_od)
        # average batch values.
        loss = torch.mean(1.0 / (image_dim - patch_size) * loss)

        test_losses.append(loss.data.item())

    print("Test epoch average loss: %f" % (np.mean(test_losses)))

    return np.mean(test_losses)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    training_loss = np.empty([seeds, epochs])
    validation_loss = np.empty([seeds, epochs])
    #NegLL_loss = np.empty([seeds, epochs])
    
    testing_loss = []
    #NLL_test_loss = []
    
    for seed in range(seeds):
        seed = seed + 1
        #seed = 5
        # Set reproducibility.
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
        # load the dataset: MNIST, Fashion-MNIST, and Omniglot.
        transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2),StochasticBinarization()])

        if dataset_name == "MNIST":
            print("loading mnist")

            train_dataset = datasets.MNIST(
                root="dataset/", train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(
                root="dataset/", train=False, transform=transform, download=True)

            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,[50000,10000])

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(dataset=valid_dataset,
                                     batch_size=batch_size, shuffle=True)
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

        # construct model and transfer to GPU
        model = ConvNADEBernoulli(D, ordering_type, num_orderings)
        print("number of model parameters:", sum(
            [np.prod(p.size()) for p in model.parameters()]))
        model = model.to(device)

        # set up the optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=gamma)
        # scheduler.step()

        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = None
        
        train_loss = []
        valid_loss = []
        #NLL_loss = []
        
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
            #NLL_loss.append(NLL(valid_loader))
            
            # #### Save Model ####
            # if checkpoint_save_name and (epoch+1) % 10 == 0:
            #     checkpoint_name = "model/model%d_%d.tar" % (epoch+1,ordering_type)
            #     save_model(epoch, model, optimizer,
            #                training_loss, testing_loss, filepath=checkpoint_name)    

        print("optimization done. full test set eval:")
        
        training_loss[seed-1, :] = train_loss
        validation_loss[seed-1, :] = valid_loss
        #NegLL_loss[seed-1, :] = NLL_loss
        
        loss = test_run(test_loader)
        testing_loss.append(loss)
        
        print(loss)

        #loss = NLL(test_loader)
        #NLL_test_loss.append(loss)
        
        #print(loss)

    #### Generate Images ####
    (data, _) = next(iter(test_loader))
    data = data.view(batch_size, 1, D, D).to(device=device)
    generate_image_orig_samples(data[:100])
    generate_recon_compar(data[:100])
    #generate_gen_compar(100,512)
    
    # train_rand_mean = np.mean(training_loss_Rand, axis=0)
    # train_rand_std = np.std(training_loss_Rand, axis=0) / np.sqrt(5)
    
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
    # test_rand_std = np.std(testing_loss_Rand, axis=0) / np.sqrt(5)

    # plt.plot(list(range(1,epochs+1)), test_rand_mean, label="Random Validation Loss", marker='o', linestyle='-', linewidth=1.5) 
    # plt.fill_between(range(epochs),test_rand_mean-2*test_rand_std, test_rand_mean+2*test_rand_std, color='red', alpha=0.2)

    # plt.plot(list(range(1,epochs+1)), testing_loss_LD, label='LD Validation Loss', marker='v', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), testing_loss_Rastor, label='Rastor Validation Loss', marker='v', linestyle='-', linewidth=1.5)

    # plt.xlabel('Epoch'); plt.ylabel('Loss') 

    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.legend(fontsize=10, frameon=False)
    # plt.tight_layout()
    # plt.show()
    
    # NLL_loss_rand_mean = np.mean(NLL_loss_Rand, axis=0)
    # NLL_loss_rand_std = np.std(NLL_loss_Rand, axis=0) / np.sqrt(5)

    # plt.plot(list(range(1,epochs+1)), NLL_loss_rand_mean, label="Random NLL Loss", marker='o', linestyle='-', linewidth=1.5) 
    # plt.fill_between(range(epochs),NLL_loss_rand_mean-2*NLL_loss_rand_std, NLL_loss_rand_mean+2*NLL_loss_rand_std, color='red', alpha=0.2)

    # plt.plot(list(range(1,epochs+1)), NLL_loss_LD, label='LD NLL Loss', marker='v', linestyle='-', linewidth=1.5)
    # plt.plot(list(range(1,epochs+1)), NLL_loss_Rastor, label='Rastor NLL Loss', marker='v', linestyle='-', linewidth=1.5)

    # plt.xlabel('Epoch'); plt.ylabel('Loss') 

    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.legend(fontsize=10, frameon=False)
    # plt.tight_layout()
    # plt.show()
