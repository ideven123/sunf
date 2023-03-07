"""
This module contains all the key-based transformation for adversarial defense.
"""

import torch
import torch.nn as nn
import numpy as np


__all__ = ["Shuffle", "NP", "FFX"]


class DataTransform(nn.Module):
    """
    Generic class for block-wise transformation.
    """

    def __init__(self ):
        super().__init__()
        self.block_size = 4
        self.blocks_axis0 = int(32 / 4)
        self.blocks_axis1 = int(32 /4)
        self.mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.Tensor([0.2471, 0.2435, 0.2616])

    def segment(self, X):
        X = X.permute(0, 2, 3, 1)
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.block_size,
            self.blocks_axis1,
            self.block_size,
            3,
        )
        X = X.permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size * self.block_size * 3,
        )
        return X

    def integrate(self, X):
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size,
            self.block_size,
            3,
        )
        X = X.permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(
            -1,
            self.blocks_axis0 * self.block_size,
            self.blocks_axis1 * self.block_size,
            3,
        )
        X = X.permute(0, 3, 1, 2)
        return X

    def generate_key(self, seed, binary=False):
        torch.manual_seed(seed)
        key = torch.randperm(self.block_size * self.block_size * 3)
        if binary:
            key = key > len(key) / 2
        return key

    def normalize(self, X):
        return (X - self.mean.type_as(X)[None, :, None, None]) / self.std.type_as(X)[
            None, :, None, None
        ]

    def denormalize(self, X):
        return (X * self.std.type_as(X)[None, :, None, None]) + self.mean.type_as(X)[
            None, :, None, None
        ]

    def forward(self, X, decrypt=False):
        raise NotImplementedError


class Modular(DataTransform):
    def __init__(self,seed,b,mask):
        super().__init__()
        np.random.seed(seed)
        self.noise_lowerbound = 67
        self.noise_upperbound = b
        self.size = [3,32,32]
        self.p = mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if b == 0:
            self.random_noise = torch.zeros(size = self.size ).to(device)
        else:
            self.random_noise = torch.tensor( np.random.randint(low=self.noise_lowerbound,high=self.noise_upperbound,size = self.size,dtype=np.uint8)).to(device)
        self.mask = torch.tensor(np.random.choice([0,1], size=self.size, p = [1-self.p, self.p])).to(device)

    def forward(self, X):
        # mod = 256/255
        # new_data = (self.normed_nosie + data) % (mod) 
        # return new_data 
        # print(X.shape,self.random_noise.shape,self.mask.shape )
        data1 =  X * 255       
        x0 = torch.trunc(data1)
        x1 = data1 - x0
        x2 = x0 + self.random_noise
        x0_1 = (x2) % 256
        return torch.clamp((x1 + x0_1)/255, min=0, max=1) * self.mask +  X * (1 - self.mask)



# class NP(DataTransform):
#     def __init__(self, config):
#         super().__init__(config)
#         self.key = self.generate_key(config.seed, binary=True)

#     def forward(self, X, decrypt=False):
#         # uncomment the following during training
#         # X = self.denormalize(X)
#         X = self.segment(X)
#         X[:, :, :, self.key] = 1 - X[:, :, :, self.key]
#         X = self.integrate(X)
#         # uncomment the following during training
#         # X = self.normalize(X)
#         return X.contiguous()

