import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()

import torchvision

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import torch.optim as optim


class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, batch_size=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.switches = []
        self.org_shapes = []
        self.unpool_shapes = []
        self.feature_maps = []
        
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        self.linear = nn.ModuleList()
        self.drop = nn.ModuleList()
        
        ## Layer 1
        self.conv.append(nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=7, stride=2, padding=1))
        self.pool.append(nn.MaxPool2d(kernel_size=3, stride=2, padding = 1, return_indices=True))
        # self.norm.append(nn.BatchNorm2d(num_features=96))
        self.norm.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0))
        
        ## Layer 2
        self.conv.append(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=0))
        self.pool.append(nn.MaxPool2d(kernel_size=3, stride=2, padding = 1, return_indices=True))
        # self.norm.append(nn.BatchNorm2d(num_features=256))
        self.norm.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0))
        
        ## Layer 3
        self.conv.append(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1))
        self.pool.append(nn.Identity())
        self.norm.append(nn.Identity())

        ## Layer 4
        self.conv.append(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1))
        self.pool.append(nn.Identity())
        self.norm.append(nn.Identity())
        
        ## Layer 5
        self.conv.append(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.pool.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True))
        self.norm.append(nn.Identity())
        
        ## Layer 6
        self.linear.append(nn.Linear(9216, 4096))
        self.drop.append(nn.Dropout(p=0.5))
        
        ## Layer 7
        self.linear.append(nn.Linear(4096, 4096))
        self.drop.append(nn.Dropout(p=0.5))
        
        ## Output
        self.linear.append(nn.Linear(4096, self.out_channels))
        self.drop.append(nn.Identity())
        
        # ## Initialize weights
        # self.apply(self._init_weights_1)
        self._init_weights_2()
    
#     def _init_weights_1(self, module):
#         if isinstance(module, nn.Conv2d):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1e-2)

#         if isinstance(module, nn.BatchNorm2d):
#             # BatchNorm with a mean of 0 = bias and a variance of 1 = weight:
#             module.bias.data.zero_()
#             module.weight.data.fill_(1e-2)

#         elif isinstance(module, nn.Linear):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1e-2)
    
    def _init_weights_2(self):
        
        for M in self.modules():
            if isinstance(M, nn.Conv2d):
                # nn.init.kaiming_uniform_(M.weight)
                nn.init.constant_(M.weight, 1e-2)

                if M.bias is not None:
                    nn.init.constant_(M.bias, 0)

            elif isinstance(M, nn.BatchNorm2d):
                # BatchNorm with a mean of 0 = bias and a variance of 1e-2 = weight:
                nn.init.constant_(M.weight, 1e-2)
                
                if M.bias is not None:
                    nn.init.constant_(M.bias, 0)

            elif isinstance(M, nn.Linear):
                # nn.init.kaiming_uniform_(M.weight)
                nn.init.constant_(M.weight, 1e-2)
                
                if M.bias is not None:
                    nn.init.constant_(M.bias, 0)
    
    def forward(self, x):
        
        for i in range(len(self.conv)):
            
            self.org_shapes.append(x.shape)
            
            x = self.conv[i](x)
            x = F.relu(x)
            
            if isinstance(self.pool[i], nn.MaxPool2d):
                self.unpool_shapes.append(x.shape)
                x, indices = self.pool[i](x)
                self.switches.append(indices)
                
            else:
                self.unpool_shapes.append(None)
                x = self.pool[i](x)
                self.switches.append(None)
                
            ## Local Contrast Normalization across feature maps (similar to AlexNet)
            x = self.norm[i](x)
            
            self.feature_maps.append(x)
            
        ## Flatten tensor for Linear Layers
        
        x = torch.flatten(x, 1)
        
        
        for i in range(len(self.linear) - 1):
            x = self.linear[i](x)
            x = self.drop[i](x)
            x = F.relu(x)
            
        x = self.linear[-1](x)
        x = F.softmax(x, dim=1)
        
        return x
    

class DeConvNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.conv_model = model
        
        self.deconv = nn.ModuleList()
        self.unpool = nn.ModuleList()
        
        ## Layer 1
        # Check if we need to set `out_padding` needs to be set
        self.deconv.append(nn.ConvTranspose2d(in_channels=96, out_channels=self.conv_model.in_channels, kernel_size=7, stride=2, padding=1, output_padding=1))
        self.unpool.append(nn.MaxUnpool2d(kernel_size=3, stride=2, padding = 0))
        
        ## Layer 2
        # Check if we need to set `out_padding` needs to be set
        self.deconv.append(nn.ConvTranspose2d(in_channels=256, out_channels=96, kernel_size=5, stride=2, padding=0, output_padding=0))
        self.unpool.append(nn.MaxUnpool2d(kernel_size=3, stride=2, padding = 0))
        
        ## Layer 3
        # Check if we need to set `out_padding` needs to be set
        self.deconv.append(nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, output_padding=0))
        self.unpool.append(nn.Identity())

        ## Layer 4
        # Check if we need to set `out_padding` needs to be set
        self.deconv.append(nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, output_padding=0))
        self.unpool.append(nn.Identity())
        
        ## Layer 5
        # Check if we need to set `out_padding` needs to be set
        self.deconv.append(nn.ConvTranspose2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, output_padding=0))
        self.unpool.append(nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0))
        
        self._init_weights_2()
        
    def _init_weights_2(self):
        for i in range(len(self.deconv)):
            with torch.no_grad():
                self.deconv[i].weight.copy_(self.conv_model.conv[i].weight)
                # self.deconv[i].weight = self.conv_model.conv[i].weight
                # self.deconv[i].weight = nn.Parameter(self.conv_model.conv[i].weight.detach().clone())
                # self.deconv[i].weight = nn.Parameter(torch.empty_like(self.conv_model.conv[i].weight).copy_(self.conv_model.conv[i].weight))
                
    
    def forward(self, layer_no=1):
            
        idx = layer_no - 1
        x = self.conv_model.feature_maps[idx].detach().clone()
        
        ## Set all Activation except target activation position as 0
        temp = x[..., 0, 0].clone()
        x.fill_(0)
        x[..., 0, 0] = temp
        
        for i in range(idx, -1, -1):
            
            if self.conv_model.switches[i] is not None:
                x = self.unpool[i](x, self.conv_model.switches[i], output_size=self.conv_model.unpool_shapes[i])
                
            x = F.relu(x)
            
            x = self.deconv[i](x)
            
            ## Just to make sure that this code runs for other image shapes (i.e., other than 224 x 224)
            ## BUT its better to change the model paddings accordingly without using this
            if x.shape != self.conv_model.org_shapes[i]:
                # skipping batch_size and no._of_channels
                x = TF.resize(x, size=self.conv_model.org_shapes[i][2:])
                
        return x