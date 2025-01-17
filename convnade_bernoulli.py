"""Implementation of Convolutional Neural Autoregressive Distribution Estimator (ConvNADE) [1].
"""

import numpy as np
import torch
from torch import nn

import qmcpy as qp


class ConvNADEBernoulli(nn.Module):
    
    """The Convolutional Neural Autoregressive Distribution Estimator (ConvNADE) model."""

    def __init__(self, NX, ordering_type, num_orderings, sample_fn=None):
        """Initializes a new ConvNADE instance.

        Args:
            width: The width of images.
            height: The height of images.
            hidden sizes: a list of integers; number of units in hidden layers
            sample_fn: See the base class.
        """
        super().__init__()
        self.width = NX
        self.height = NX
        self._input_dim = NX * NX
        self.num_orderings = num_orderings
        
        self.orderings = []
        
        pixel_ordering = np.arange(self._input_dim)
        
        # Select ordering of pixels.
        if ordering_type == 1:
            pixel_ordering = np.random.permutation(self._input_dim)
                
        elif ordering_type == 2: # low-dis ordering 
            sobol_gen = qp.Sobol(2, randomize=False)# Try No Scrambling. 
            sobol_points = sobol_gen.gen_samples(n=self._input_dim)
                
            pixel_ordering = (np.floor(sobol_points * NX)[:,0] + (NX-1 - np.floor(sobol_points * NX)[:,1])*NX).astype(int)
        
        self.orderings.append(pixel_ordering)
        
        # define a simple Convolutional neural net
        self.net = []
        
        # Input Layer.
        self.net.append(nn.Conv2d(2, 64, kernel_size= (8,8), stride=1, padding='valid'))
        self.net.append(nn.ReLU())
        
        # Hidden Layers.
        self.net.append(nn.Conv2d(64, 64, kernel_size= (4,4), stride=1, padding='valid'))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(64, 64, kernel_size= (6,6), stride=1, padding='valid'))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(64, 64, kernel_size= (7,7), stride=1, padding='valid'))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(64, 64, kernel_size= (7,7), stride=1, padding=6))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(64, 64, kernel_size= (6,6), stride=1, padding=5))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(64, 64, kernel_size= (4,4), stride=1, padding=3))
        self.net.append(nn.ReLU())
        
        # Output Layer.
        self.net.append(nn.Conv2d(64, 1, kernel_size= (8,8), stride=1, padding=7))
        self.net = nn.Sequential(*self.net)
        
        self._create_shape_buffers(1, self.width, self.height)

    def load_state_dict(self, state_dict, strict=True):
        """Registers dynamic buffers before loading the model state."""
        if "_c" in state_dict and not getattr(self, "_c", None):
            c, h, w = state_dict["_c"], state_dict["_h"], state_dict["_w"]
            self._create_shape_buffers(c, h, w)
        super().load_state_dict(state_dict, strict)

    def _create_shape_buffers(self, channels, height, width):
        channels = channels if torch.is_tensor(channels) else torch.tensor(channels)
        height = height if torch.is_tensor(height) else torch.tensor(height)
        width = width if torch.is_tensor(width) else torch.tensor(width)
        self.register_buffer("_c", channels)
        self.register_buffer("_h", height)
        self.register_buffer("_w", width)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        """Computes the forward pass.

        Args:
            x: Either a tensor of images with shape (n, 1, h, w).
        Returns:
            The result of the forward pass.
        """
        return self.net(x) # Returns conditional distribution values.