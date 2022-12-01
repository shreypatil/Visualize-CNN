# Invariance and Occlusion Sensitivity Plots

import torch
from torch import Tensor
import numpy as np

def euclidean_distance(vec1: Tensor, vec2: Tensor):
    # dim=None flattens the vector to find the euclidean distance
    return torch.linalg.vector_norm(vec1 - vec2, ord=2, dim=None)

def euclidean_distance_test():
    
    vec1 = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    vec2 = torch.zeros_like(vec1, dtype=torch.float32)

    print(euclidean_distance(vec1, vec1) == 0)
    print(euclidean_distance(vec1, vec2) == np.sqrt(np.sum(np.arange(9)**2)))

def invariance_plots():
    pass

def occlusion_sensitivity():
    pass

def main():
    euclidean_distance_test()

if __name__ == "__main__":
    main()