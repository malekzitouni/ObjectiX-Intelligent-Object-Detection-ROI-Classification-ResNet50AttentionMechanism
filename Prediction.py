

import time

import numpy as np
import ResAttention as Net
import os
import matplotlib.pyplot as plt
import torch
import GetCOCOCatNames
import numpy as np
import cv2
#...........................................Input Parameters.................................................

Trained_model_path=r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\logs\1100_weights.torch" # Pretrained model
ImageFile=r'C:\Users\Pc\Desktop\COCO_Dataset\TestImages\Test2\Image.png' #Input image
ROIMaskFile= r'C:\Users\Pc\Desktop\COCO_Dataset\TestImages\Test2\InputMask1.png' # Input ROI mas
UseCuda=True
#---------------------Get list of coco classes-----------------------------------------------------------------------------
CatNames=GetCOCOCatNames.GetCOCOCatNames()
#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=CatNames.__len__())


Net.load_state_dict(torch.load(Trained_model_path)) #Load net
if UseCuda: Net.cuda()
Net.eval()
#--------------------Read Image and segment mask---------------------------------------------------------------------------------
Images=cv2.imread(ImageFile)
ROIMask=cv2.imread(ROIMaskFile,0)
plt.imshow(Images)
plt.savefig('Images_output.png')  # Save image instead of displaying
plt.imshow(ROIMask * 255)  # Display ROI mask
plt.savefig('ROIMask_output.png')
Images=np.expand_dims(Images,axis=0)
ROIMask=np.expand_dims(ROIMask,axis=0)
#-------------------Run Prediction----------------------------------------------------------------------------
Prob, PredLb = Net.forward(Images, ROI=ROIMask)  # Run net inference and get prediction
PredLb = PredLb.data.cpu().numpy()
Prob = Prob.data.cpu().numpy()
#---------------Print Prediction on screen--------------------------------------------------------------------------
print("Predicted Label " + CatNames[PredLb[0]])
print("Predicted Label Prob="+str(Prob[0,PredLb[0]]*100)+"%")