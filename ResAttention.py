import scipy.misc as misc
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
        def __init__(self,NumClasses, UseGPU=True): 
            super(Net, self).__init__()
            self.UseGPU = UseGPU # Use GPu with cuda
            self.Net = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.Net.fc=nn.Linear(2048, NumClasses)

            self.Attentionlayer = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True) 
            self.Attentionlayer.bias.data = torch.ones(self.Attentionlayer.bias.data.shape)
            self.Attentionlayer.weight.data = torch.zeros(self.Attentionlayer.weight.data.shape)

        def forward(self,Images,ROI):
                      
                if isinstance(Images, np.ndarray):
                    InpImages = torch.from_numpy(Images)  # Convert numpy array to tensor
                else:
                    InpImages = Images  # If it's already a tensor, use it directly

                # Reorder dimensions to (Batch, Channels, Height, Width)
                InpImages = InpImages.permute(0, 3, 1, 2).float()  
                # Assuming ROI could be either a numpy array or a PyTorch tensor
                if isinstance(ROI, np.ndarray):
                    ROImap = torch.from_numpy(ROI.astype(float)).unsqueeze(dim=1).float()  # Convert numpy array to tensor and add new dimension
                else:
                    ROImap = ROI.unsqueeze(dim=1).float()  # If it's already a tensor, add new dimension and convert to float
                     
                if self.UseGPU == True: # Convert to GPU
                    InpImages = InpImages.cuda()
                    ROImap = ROImap.cuda()
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # Normalize image by std and mean
                x=InpImages

                x = self.Net.conv1(x) 
                AttentionMap = self.Attentionlayer(F.interpolate(ROImap, size=x.shape[2:4], mode='bilinear'))
                x = x + AttentionMap

                x = self.Net.bn1(x)
                x = self.Net.relu(x)
                x = self.Net.maxpool(x)
                x = self.Net.layer1(x)
                x = self.Net.layer2(x)
                x = self.Net.layer3(x)
                x = self.Net.layer4(x)
                x = torch.mean(torch.mean(x, dim=2), dim=2)
                x = self.Net.fc(x)
                ProbVec = F.softmax(x,dim=1) # Probability vector for all classes
                Prob,Pred=ProbVec.max(dim=1) # Top predicted class and probability
                return ProbVec,Pred