import numpy as np
import ResAttention as Net
import CocoReader as COCOReader
import os
import torch
import torch.optim as optim
from sklearn.model_selection import KFold  # Import KFold for cross-validation
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard's SummaryWriter

# Start of script
print("train.py script execution has started.", flush=True)

# Set paths and parameters
TrainImageDir = r'C:\Users\Pc\Desktop\COCO_Dataset\train2017\train2017'
TrainAnnotationFile = r'C:\Users\Pc\Desktop\COCO_Dataset\annotations_trainval2017\annotations\instances_train2017.json'
logs_dir = r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\logs"
TrainLossTxtFile = r'C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\TrainLoss.txt'

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

# Training parameters
MinSize = 160
MaxSize = 1000
MaxBatchSize = 100
MaxPixels = 800 * 800 * 8
Learning_Rate = 9.99999e-06
Weight_Decay = 1e-5
num_epochs = 1  # Set to the number of epochs you want to train
total_images = 118287
batch_size = 100
num_folds = 5  # Define the number of folds for cross-validation

# Create the dataset reader and handle any errors
try:
    Reader = COCOReader.COCOReader(TrainImageDir, TrainAnnotationFile, batch_size, MinSize, MaxSize, MaxPixels)
    print(f"Dataset reader initialized successfully with {Reader.NumCats} categories.", flush=True)
except Exception as e:
    print(f"Failed to initialize dataset reader: {e}", flush=True)
    raise e

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=logs_dir)

# Create K-Fold object
kf = KFold(n_splits=num_folds, shuffle=True)  # Shuffle the data for better cross-validation

# Initialize the network
Net = Net.Net(NumClasses=Reader.NumCats, UseGPU=UseCuda).to(device)
print("Network initialized.", flush=True)

# Set optimizer
optimizer = optim.AdamW(params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)

# Mixed precision scaler for efficiency (if GPU available)
scaler = torch.amp.GradScaler() if UseCuda else None

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(total_images))):
    print(f"Starting Fold {fold + 1}/{num_folds}", flush=True)

    # Reset model weights for each fold
    Net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # Reset optimizer
    optimizer = optim.AdamW(params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} for Fold {fold + 1}", flush=True)

        total_train_loss = 0
        # Implement the batching logic using the train_idx
        for batch in range(len(train_idx) // batch_size):
            try:
                # Read a batch of data using train_idx to read the correct samples for this fold
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
                    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 1e-7)))

                # Backpropagation and optimizer step
                optimizer.zero_grad()
                scaler.scale(Loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += Loss.item()

                # Log training loss to TensorBoard
                writer.add_scalar(f'Fold_{fold+1}/Train_Loss', Loss.item(), epoch * (len(train_idx) // batch_size) + batch)

                # Print training loss
                if (batch + 1) % 100 == 0:
                    print(f"Fold {fold + 1}, Batch {batch + 1}, Loss={float(Loss.data.cpu().numpy())}", flush=True)

            except Exception as e:
                print(f"Error during training at Fold {fold + 1}, Batch {batch + 1}: {e}", flush=True)
                raise e

        avg_train_loss = total_train_loss / (len(train_idx) // batch_size)
        writer.add_scalar(f'Fold_{fold+1}/Avg_Train_Loss', avg_train_loss, epoch)
        print(f"Epoch {epoch + 1} finished for Fold {fold + 1}, Avg Train Loss: {avg_train_loss:.4f}", flush=True)

        # Validation loop
        with torch.no_grad():  # Disable gradient calculation for validation
            Net.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0

            for batch in range(len(val_idx) // batch_size):
                try:
                    # Read a batch of validation data
                    Images, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextBatchRandom()

                    # Move tensors to GPU if available
                    Images = torch.from_numpy(Images).to(device)
                    SegmentMask = torch.from_numpy(SegmentMask).to(device)
                    LabelsOneHot = torch.from_numpy(LabelsOneHot).to(device)

                    # Forward pass
                    with torch.amp.autocast(device_type='cuda' if UseCuda else 'cpu'):
                        Prob, Lb = Net.forward(Images, ROI=SegmentMask)

                    # Cross entropy loss
                    OneHotLabels = torch.autograd.Variable(LabelsOneHot, requires_grad=False)
                    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 1e-7)))

                    val_loss += Loss.item()

                    # Calculate accuracy (if applicable)
                    predicted = torch.argmax(Prob, dim=1)
                    total += LabelsOneHot.size(0)
                    correct += (predicted == torch.argmax(LabelsOneHot, dim=1)).sum().item()

                except Exception as e:
                    print(f"Error during validation at Fold {fold + 1}, Batch {batch + 1}: {e}", flush=True)
                    raise e

            # Log validation metrics to TensorBoard
            avg_val_loss = val_loss / (len(val_idx) // batch_size)
            accuracy = correct / total if total > 0 else 0
            writer.add_scalar(f'Fold_{fold+1}/Val_Loss', avg_val_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Val_Accuracy', accuracy, epoch)
            print(f"Validation Loss for Fold {fold + 1}: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}", flush=True)

# Close the TensorBoard writer
writer.close()

print("Cross-validation training complete.", flush=True)
