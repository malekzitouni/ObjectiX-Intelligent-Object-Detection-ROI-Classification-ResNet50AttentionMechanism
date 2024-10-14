import numpy as np
import ResAttention as Net
import CocoReader as COCOReader
import os
import torch
import torch.optim as optim

# Training Metrics: cross-entropy loss, Average Loss, Learning Rate
# Start of script
with open("Loss.txt", "a") as f:
    f.write("train.py script execution has started.\n")
    f.flush()

# Set paths and parameters
TrainImageDir = r'C:\Users\Pc\Desktop\COCO_Dataset\train2017\train2017'  # Path to coco images
TrainAnnotationFile = r'C:\Users\Pc\Desktop\COCO_Dataset\annotations_trainval2017\annotations\instances_train2017.json'  # Path to coco instance annotation file
logs_dir = r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\logsR"  # Path to logs directory
TrainLossTxtFile = r'C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\Detection\Lo.txt'

# Ensure directories exist and handle any errors
try:
    os.makedirs(logs_dir, exist_ok=True)
    with open("Lo.txt", "a") as f:
        f.write(f"Logs directory created: {logs_dir}\n")
        f.flush()
except Exception as e:
    with open("Lo.txt", "a") as f:
        f.write(f"Failed to create logs directory: {e}\n")
        f.flush()
    raise e

# Set device (CUDA or CPU)
UseCuda = torch.cuda.is_available()
device = torch.device("cuda" if UseCuda else "cpu")
with open("Lo.txt", "a") as f:
    f.write(f"UseCuda: {UseCuda}, Device: {device}\n")
    f.flush()

# Debug: Print paths to ensure correctness
with open("Lo.txt", "a") as f:
    f.write(f"TrainImageDir: {TrainImageDir}\n")
    f.write(f"TrainAnnotationFile: {TrainAnnotationFile}\n")
    f.write(f"TrainLossTxtFile: {TrainLossTxtFile}\n")
    f.flush()

# Training parameters
MinSize = 160
MaxSize = 1000
MaxBatchSize = 100

MaxPixels = 8002 * 800 * 8
checkpoint_path = r'C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\Detection\checkpointt.pth'  # Path to save the checkpoint
start_epoch = 0
start_batch = 0
Learning_Rate = 9.99999e-06
learning_rate_decay = 0.999999
Weight_Decay = 1e-5
num_epochs = 1  # Set to the number of epochs you want to train
total_images = 118287
batch_size = 100
nbr_batches = total_images // batch_size

# Create the dataset reader and handle any errors
try:
    Reader = COCOReader.COCOReader(TrainImageDir, TrainAnnotationFile, batch_size, MinSize, MaxSize, MaxPixels)
    with open("Lo.txt", "a") as f:
        f.write(f"Dataset reader initialized successfully with {Reader.NumCats} categories.\n")
        f.flush()
except Exception as e:
    with open("Lo.txt", "a") as f:
        f.write(f"Failed to initialize dataset reader: {e}\n")
        f.flush()
    raise e

# Initialize the network
try:
    Net = Net.Net(NumClasses=Reader.NumCats, UseGPU=UseCuda).to(device)
    with open("Lo.txt", "a") as f:
        f.write("Network initialized.\n")
        f.flush()
except Exception as e:
    with open("Lo.txt", "a") as f:
        f.write(f"Failed to initialize network: {e}\n")
        f.flush()
    raise e

# Set optimizer
optimizer = optim.AdamW(params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)

# Load checkpoint if available
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        Net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        Learning_Rate = checkpoint['lr']
        with open("Lo.txt", "a") as f:
            f.write(f"Resuming training from epoch {start_epoch}, batch {start_batch}\n")
            f.flush()
    except Exception as e:
        with open("Lo.txt", "a") as f:
            f.write(f"Failed to load checkpoint: {e}\n")
            f.flush()
        raise e
else:
    with open("Lo.txt", "a") as f:
        f.write("No checkpoint found. Initializing a new model.\n")
        f.flush()

# Create file for saving loss
with open("Lo.txt", "a") as f:
    f.write(f"Iteration\tloss\tLearning Rate={Learning_Rate}\n")

# Mixed precision scaler for efficiency (if GPU available)
scaler = torch.amp.GradScaler() if UseCuda else None

# Training loop
AVGLoss = 0

for epoch in range(start_epoch, num_epochs):
    with open("Lo.txt", "a") as f:
        f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
        f.flush()

    for batch in range(start_batch, nbr_batches):
        try:
            # Read a batch of data
            with open("Lo.txt", "a") as f:
                f.write(f"Start of batch {batch + 1}\n")
                f.flush()

            Images, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextBatchRandom()

            # Move tensors to GPU if available
            Images = torch.from_numpy(Images).to(device)
            SegmentMask = torch.from_numpy(SegmentMask).to(device)
            LabelsOneHot = torch.from_numpy(LabelsOneHot).to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda' if UseCuda else 'cpu'):
                Prob, Lb = Net.forward(Images, ROI=SegmentMask)

                # Cross entropy loss
                OneHotLabels = torch.autograd.Variable(LabelsOneHot, requires_grad=False)
                Loss = -torch.mean((OneHotLabels * torch.log(Prob + 1e-7)))  # Cross entropy loss

                # Running average loss calculation
                AVGLoss = AVGLoss * 0.999 + 0.001 * float(Loss.data.cpu().numpy()) if AVGLoss != 0 else float(Loss.data.cpu().numpy())

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            if scaler:
                scaler.scale(Loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                Loss.backward()
                optimizer.step()

            # Save model and checkpoint every 100 batches
            if (batch + 1) % 100 == 0:
                with open("Lo.txt", "a") as f:
                    f.write(f"Saving model and checkpoint for batch {batch + 1} \n")
                    f.flush()

                # Save only the model weights
                model_weights_path = os.path.join(logs_dir, f"{batch + 1}_weightsedit.torch")
                torch.save(Net.state_dict(), model_weights_path)

                # Save a complete checkpoint including epoch, batch, and optimizer state
                checkpoint_data = {
                    'epoch': epoch,
                    'batch': batch + 1,
                    'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': Learning_Rate
                }
                torch.save(checkpoint_data, checkpoint_path)

            # Save loss to ResumeLoss.txt every 50 batches
            if (batch + 1) % 100 == 0:
                with open("trainR", "a") as f:
                    f.write(f"Batch {batch + 1} \t Loss={Loss.item()} \t AVGLoss={AVGLoss}\n")
                    f.flush()

                

        except Exception as e:
            with open("Lo.txt", "a") as f:
                f.write(f"Error during training at batch {batch + 1}: {e}\n")
                f.flush()
            raise e

    # Update learning rate with decay
    Learning_Rate *= learning_rate_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = Learning_Rate

# End of script
with open("Lo.txt", "a") as f:
    f.write("train.py script execution has finished.\n")
    f.flush()
