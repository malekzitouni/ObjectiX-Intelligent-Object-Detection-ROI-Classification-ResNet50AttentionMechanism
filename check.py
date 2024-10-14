import torch
import numpy as np
import ResAttention as Net
import torch.optim as optim

import CocoReader as COCOReader
import os
Learning_Rate = 9.99999e-06
Weight_Decay = 1e-5

# Training parameters
MinSize = 160
MaxSize = 1000
MaxBatchSize = 100
MaxPixels = 800 * 800 * 8
Trained_model_path = ""
checkpoint_path = "check.pth"
start_epoch = 0
start_batch = 0
Learning_Rate = 9.99999e-06
learning_rate_decay = 0.999999
Weight_Decay = 1e-5
num_epochs = 1  # Set to the number of epochs you want to train
total_images = 118287
batch_size = 100
nbr_batches = total_images // batch_size
TrainImageDir = r'C:\Users\Pc\Desktop\COCO_Dataset\train2017\train2017'  # Path to coco images
TrainAnnotationFile = r'C:\Users\Pc\Desktop\COCO_Dataset\annotations_trainval2017\annotations\instances_train2017.json'  # Path to coco instance annotation file
logs_dir = r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\logs"  # Path to logs directory
# Set device (CUDA or CPU)
UseCuda = torch.cuda.is_available()
device = torch.device("cuda" if UseCuda else "cpu")
print(f"UseCuda: {UseCuda}, Device: {device}", flush=True)
# Create the dataset reader and handle any errors
try:
    Reader = COCOReader.COCOReader(TrainImageDir, TrainAnnotationFile, batch_size, MinSize, MaxSize, MaxPixels)
    print(f"Dataset reader initialized successfully with {Reader.NumCats} categories.", flush=True)
except Exception as e:
    print(f"Failed to initialize dataset reader: {e}", flush=True)
    raise e

# Initialize the network
try:
    Net = Net.Net(NumClasses=Reader.NumCats, UseGPU=UseCuda).to(device)
    print("Network initialized.", flush=True)
except Exception as e:
    print(f"Failed to initialize network: {e}", flush=True)
    raise e

# Set optimizer
optimizer = optim.AdamW(params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)

# Assuming 'Net' is your model and 'optimizer' is your optimizer
# Load the pre-trained weights
pretrained_weights_path = r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\logs\1100_weights.torch"  # Change this to the correct path
Net.load_state_dict(torch.load(pretrained_weights_path,weights_only=True))

# Create a checkpoint
checkpoint_data = {
    'model_state_dict': Net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr': Learning_Rate,  # Your current learning rate
}

torch.save(checkpoint_data, 'check.pth')


#torch.load(pretrained_weights_path): This loads the weights (or the entire checkpoint, depending on how the file was saved) from the file specified by pretrained_weights_path. This typically returns a dictionary of tensors containing the model parameters.
#Net.load_state_dict(): This method takes the loaded state dictionary (which contains the model weights) and assigns them to the model Net. This updates the model's parameters to the pre-trained weights