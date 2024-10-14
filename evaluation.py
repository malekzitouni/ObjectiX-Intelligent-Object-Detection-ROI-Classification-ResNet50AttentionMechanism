import numpy as np
import ResAttention as Net
import CocoReader as COCOReader
import os
import torch
import matplotlib.pyplot as plt  # For plotting
import torch.optim as optim


# ...........................................Input Parameters.................................................

checkpoint_path = r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\Detection\check.pth"
TestImageDir = r"C:\Users\Pc\Desktop\COCO_Dataset\val2017\val2017"
TestAnnotationFile = r"C:\Users\Pc\Desktop\COCO_Dataset\annotations_trainval2017\annotations\instances_val2017.json"
ValidationLossFile = r"C:\Users\Pc\Downloads\-ObjectiX-Intelligent-Object-Detection-ROI-Classification--main\Validation.txt"
SamplePerClass = 2
UseCuda = torch.cuda.is_available()
EvaluationFile = "VALL.txt"
Learning_Rate = 9.99999e-06
learning_rate_decay = 0.999999
Weight_Decay = 1e-5

# Initialize the COCOReader and the network
Reader = COCOReader.COCOReader(TestImageDir, TestAnnotationFile)
NumClasses = Reader.NumCats
device = torch.device("cuda" if UseCuda else "cpu")
Net = Net.Net(NumClasses=NumClasses, UseGPU=UseCuda)

# Move model to the appropriate device
if UseCuda:
    Net = Net.cuda()

# Load the model weights
checkpoint = torch.load(checkpoint_path, map_location=device)
Net.load_state_dict(checkpoint['model_state_dict'])
Learning_Rate = checkpoint['lr']
print(Learning_Rate)
print(learning_rate_decay)
# Set the model to evaluation mode
Net.eval()

# Initialize the optimizer
optimizer = optim.AdamW(params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Sizes of detected objects
Sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 500000, 1000000]
NumSizes = len(Sizes)

# Initialize true/false positive/negative arrays
TP = np.zeros([Reader.NumCats + 1], dtype=np.float64)  # True positives per class
FP = np.zeros([Reader.NumCats + 1], dtype=np.float64)  # False positives per class
FN = np.zeros([Reader.NumCats + 1], dtype=np.float64)  # False negatives per class
SumPred = np.zeros([Reader.NumCats + 1], dtype=np.float64)  # Sum of predictions per class

SzTP = np.zeros([Reader.NumCats + 1, NumSizes], dtype=np.float64)  # True positives by size
SzFP = np.zeros([Reader.NumCats + 1, NumSizes], dtype=np.float64)  # False positives by size
SzFN = np.zeros([Reader.NumCats + 1, NumSizes], dtype=np.float64)  # False negatives by size
SzSumPred = np.zeros([Reader.NumCats + 1, NumSizes], dtype=np.float64)

# Class prediction counters
CorCatPred = np.zeros([Reader.NumCats], dtype=np.float64)  # Correct class predictions
TotalCat = np.zeros([Reader.NumCats], dtype=np.float64)

# Initialize the batch counter
batch_counter = 0

# Loss function (customize as per your model requirements)
loss_fn = torch.nn.CrossEntropyLoss()

def calculate_loss(predicted, target):
    return loss_fn(predicted, target)

# Main evaluation loop
def evaluate():
    global batch_counter
    batch_numbers = []
    batch_losses = []

    # Open the ValidationLossFile to append loss values for each batch
    with open(ValidationLossFile, "a") as loss_file:

        for c in range(Reader.NumCats):
            print(f"Class {c}) {Reader.CatNames[c]} - Num Cases: {np.min((SamplePerClass, len(Reader.ImgIds[c])))}")
            for m in range(np.min((SamplePerClass, len(Reader.ImgIds[c])))):
                Images, SegmentMask, Labels, LabelsOneHot = Reader.ReadSingleImageAndClass(ClassNum=c, ImgNum=m)

                if UseCuda:
                    Images = torch.from_numpy(Images).float().cuda()
                    SegmentMask = torch.from_numpy(SegmentMask).float().cuda()
                    Labels = torch.from_numpy(Labels).long().cuda()

                print(f"Class {c}) {Reader.CatNames[c]}  {m}")
                BatchSize = Images.shape[0]

                for i in range(BatchSize):
                    # Inference with mixed precision (if CUDA is used)
                    with torch.cuda.amp.autocast(enabled=UseCuda):
                        Prob, Lb = Net.forward(Images[i:i + 1], ROI=SegmentMask[i:i + 1])

                    # Ensure predicted probabilities are the right shape
                    if Prob.dim() == 1:  # In case Prob is returned as a 1D tensor, reshape it to 2D
                        Prob = Prob.unsqueeze(0)

                    PredLb = Prob.argmax(dim=1).cpu().numpy()  # Get predicted label (highest probability)

                    # Calculate loss
                    loss = calculate_loss(Prob, Labels[i:i+1])  # Define your loss function

                    # Collect loss for plotting
                    batch_numbers.append(batch_counter)
                    batch_losses.append(loss.item())

                    # Write loss for current batch to the validation loss file
                    loss_file.write(f"Batch {batch_counter}, Class {Reader.CatNames[c]}, Image {m}, Loss: {loss.item()}\n")

                    # Increment batch counter
                    batch_counter += 1

                    # Get the size of the labeled region in the image
                    LbSize = SegmentMask[i].sum()
                    SzInd = next((f for f, sz in enumerate(Sizes) if LbSize < sz), -1)

                    # True positive: correct class predicted
                    if PredLb[0] == Labels[i]:
                        TP[Labels[i]] += 1
                        SzTP[Labels[i], SzInd] += 1
                    else:
                        # False negative and false positive
                        FN[Labels[i]] += 1
                        FP[PredLb[0]] += 1
                        SzFN[Labels[i], SzInd] += 1
                        SzFP[PredLb[0], SzInd] += 1

                    # Update sum of predictions
                    SumPred[Labels[i]] += 1
                    SzSumPred[Labels[i], SzInd] += 1

    return batch_numbers, batch_losses

# Call the evaluate function
batch_numbers, batch_losses = evaluate()

# Plotting the loss values after evaluation
plt.figure()
plt.plot(batch_numbers, batch_losses, 'r-', label='Loss')
plt.title('Loss During Evaluation')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_during_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()

# Writing evaluation results
with open(EvaluationFile, "w") as f:
    NrmF = len(SumPred) / (np.sum(SumPred > 0))  # Normalization factor for classes with zero occurrences

    # Accuracy calculations (per class and overall)
    accuracy = TP / (TP + FP + FN + 1e-8)
    mean_acc_all = accuracy.mean() * NrmF * 100
    mean_acc_img = (TP.mean() / SumPred.mean()) * 100

    # Precision, Recall, F1-score, and False Positive Rate (FPR)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    fpr = FP / (FP + (SumPred - (TP + FP + FN)) + 1e-8)  # Approximation for FPR

    # Writing global metrics to file
    f.write(f"Mean Accuracy All Class Average =\t{mean_acc_all}%\r\n")
    f.write(f"Mean Accuracy Images =\t{mean_acc_img}%\r\n")
    f.write(f"Mean Precision =\t{precision.mean() * 100}%\r\n")
    f.write(f"Mean Recall =\t{recall.mean() * 100}%\r\n")
    f.write(f"Mean F1 Score =\t{f1_score.mean() * 100}%\r\n")
    f.write(f"Mean FPR =\t{fpr.mean() * 100}%\r\n")

    print("\n=============================================================================\n")
    f.write("SizeMax\tMeanClasses\tMeanGlobal\tNum Instances\tNumValidClasses\tPrecision\tRecall\tF1\tFPR\r\n")

    for i, sz in enumerate(Sizes):
        if SzSumPred[:, i].sum() == 0: 
            continue
        NumValidClass = np.sum(SzSumPred[:, i] > 0)  # Valid classes with predictions
        f.write(f"{sz}\t{TP[i] / SumPred[i] * 100}\t{TP[i] / (TP[i] + FP[i] + 1e-8) * 100}\t{SzSumPred[:, i].sum()}\t{NumValidClass}\t{precision.mean() * 100}\t{recall.mean() * 100}\t{f1_score.mean() * 100}\t{fpr.mean() * 100}\r\n")
