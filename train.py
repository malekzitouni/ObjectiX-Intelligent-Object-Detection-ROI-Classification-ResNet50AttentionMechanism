import numpy as np
import ResAttention as Net
import CocoReader as COCOReader
import os
import torch
import torch.optim as optim

# Start of script
print("train.py script execution has started.", flush=True)

# Set paths and parameters
TrainImageDir = r'C:\Users\Pc\Desktop\COCO_Dataset\train2017\train2017'  # Path to coco images
TrainAnnotationFile = r'C:\Users\Pc\Desktop\COCO_Dataset\annotations_trainval2017\annotations\instances_train2017.json'  # Path to coco instance annotation file
logs_dir = r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\logs"  # Path to logs directory
TrainLossTxtFile = r'C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\TrainLoss.txt'
ValidLossTxtFile = r'C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\ValidationLoss.txt'

# Ensure directories exist and handle any errors
try:
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs directory created: {logs_dir}", flush=True)
except Exception as e:
    print(f"Failed to create logs directory: {e}", flush=True)
    raise e

# Set device (CUDA or CPU)
UseCuda = torch.cuda.is_available()
device = torch.device("cuda" if UseCuda else "cpu")
print(f"UseCuda: {UseCuda}, Device: {device}", flush=True)

# Debug: Print paths to ensure correctness
print(f"TrainImageDir: {TrainImageDir}", flush=True)
print(f"TrainAnnotationFile: {TrainAnnotationFile}", flush=True)
print(f"TrainLossTxtFile: {TrainLossTxtFile}", flush=True)
print(f"ValidLossTxtFile: {ValidLossTxtFile}", flush=True)

# Training parameters
MinSize = 160
MaxSize = 1000
MaxBatchSize = 100
MaxPixels = 800 * 800 * 8
Trained_model_path = ""
checkpoint_path = "checkpoint.pth"
start_epoch = 0
start_batch = 0
Learning_Rate = 1e-5
learning_rate_decay = 0.999999
Weight_Decay = 1e-5
num_epochs = 10
total_images = 118287
batch_size = 100
nbr_batches = total_images // batch_size

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

# Load pretrained model if available
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        Net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        Learning_Rate = checkpoint['lr']
        print(f"Resuming training from epoch {start_epoch}, batch {start_batch}", flush=True)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}", flush=True)
        raise e
else:
    if Trained_model_path:
        try:
            print(f"Loading pretrained model from: {Trained_model_path}", flush=True)
            Net.load_state_dict(torch.load(Trained_model_path, map_location=device))
        except Exception as e:
            print(f"Failed to load pretrained model: {e}", flush=True)
            raise e

# Create file for saving loss
with open(TrainLossTxtFile, "a") as f:
    f.write(f"Iteration\tloss\tLearning Rate={Learning_Rate}")

# Mixed precision scaler for efficiency (if GPU available)
scaler = torch.cuda.amp.GradScaler() if UseCuda else None

# Training loop
AVGLoss = 0

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}", flush=True)

    for batch in range(start_batch, nbr_batches):
        try:
            # Read a batch of data
            Images, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextBatchRandom()

            # Move tensors to GPU if available
            Images = torch.from_numpy(Images).to(device)
            SegmentMask = torch.from_numpy(SegmentMask).to(device)
            LabelsOneHot = torch.from_numpy(LabelsOneHot).to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=UseCuda):
                Prob, Lb = Net.forward(Images, ROI=SegmentMask)

                # Cross entropy loss
                OneHotLabels = torch.autograd.Variable(LabelsOneHot, requires_grad=False)
                Loss = -torch.mean((OneHotLabels * torch.log(Prob + 1e-7)))  # Cross entropy loss

                # Running average loss calculation
                AVGLoss = AVGLoss * 0.999 + 0.001 * float(Loss.data.cpu().numpy()) if AVGLoss != 0 else float(Loss.data.cpu().numpy())

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            scaler.scale(Loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Save model and checkpoint every 100 batches
            if (batch + 1) % 100 == 0:
                print(f"Saving model and checkpoint for batch {batch + 1}", flush=True)
                torch.save(Net.state_dict(), os.path.join(logs_dir, f"{batch + 1}.torch"))
                torch.save({
                    'epoch': epoch,
                    'batch': batch + 1,
                    'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': Learning_Rate
                }, checkpoint_path)

            # Log and print training loss every 100 batches
            if (batch + 1) % 100 == 0:
                print(f"Batch {batch + 1}/{nbr_batches}, Train Loss={float(Loss.data.cpu().numpy())}, Running Average Loss={AVGLoss}", flush=True)
                with open(TrainLossTxtFile, "a") as f:
                    f.write(f"\n{batch + 1}\t{float(Loss.data.cpu().numpy())}\t{AVGLoss}")

        except Exception as e:
            print(f"Error during training at batch {batch + 1}: {e}", flush=True)
            raise e

    # Reset start_batch for next epoch
    start_batch = 0

    # Decay learning rate at the end of each epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] *= learning_rate_decay

    print(f"Epoch {epoch + 1} finished, learning rate is now {optimizer.param_groups[0]['lr']}", flush=True)

print("Training complete.", flush=True)
