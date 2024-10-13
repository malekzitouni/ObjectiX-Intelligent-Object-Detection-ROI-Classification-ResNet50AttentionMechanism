# malekzitouni--ObjectiX-Intelligent-Object-Detection-ROI-Classification-

Here's a summary of the tasks you've completed during your project:

### **Project Title**: Object Detection Model Training with COCO Dataset (ObjectiX-Intelligent-Object-Detection-ROI-Classification)

#### **Model Architecture & Dataset**
- **Backbone**: ResNet-50 with attention mechanism (ROI-based object detection).
- **Dataset**: COCO Dataset (train2017, test2017).
- **Framework**: PyTorch 2.4.1
- **Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU, CUDA version 12.4.
- **Custom Loss Function**: Cross-entropy loss for object classification and ROI detection.

#### **Tasks Completed**:

1. **Setting up the Environment**:
   - Configured **Vertex AI** instance without GPU for early testing.
   - Transitioned to a local environment with **CUDA support** to utilize the **NVIDIA GeForce RTX 3050** GPU for training.

2. **Data Preparation**:
   - Downloaded and processed the **COCO dataset**.
   - Used **COCOReader** for efficient reading of images and annotations.
   - Applied Region of Interest (ROI) masking for specific image region classification.

3. **Model Training**:
   - Trained the object detection model from scratch using **ResNet-50** as the backbone.
   - Training process involved:
     - **Single Epoch Training**: Due to GPU limitations, trained the model for only 1 epoch on a large 300 million parameter model.
     - Saved model checkpoints and logs at intervals for tracking.

4. **Evaluation and Metrics**:
   - Evaluated the model after 1 epoch of training, achieving the following metrics:
     - **Mean Accuracy All Class Average**: 30.05%
     - **Mean Accuracy Images**: 47.69%
     - **Mean Precision**: 38.20%
     - **Mean Recall**: 38.22%
     - **Mean F1 Score**: 35.24%

5. **Use of Pre-trained Model**:
   - Downloaded and fine-tuned a **pre-trained model** from [this repository](https://github.com/sagieppel/Classification-of-object-in-a-specific-image-region-using-a-convolutional-neural-net-with-ROI-mask-a).
   - The pre-trained model was trained for 10 epochs, and its evaluation yielded much better performance:
     - **Mean Accuracy All Class Average**: 70.13%
     - **Mean Accuracy Images**: 77.85%
     - **Mean Precision**: 81.90%
     - **Mean Recall**: 79.43%
     - **Mean F1 Score**: 77.98%

6. **Retraining the Pre-trained Model**:
   - Resumed training using the saved checkpoint `88000.torch` from the pre-trained model.
   - After retraining, the performance slightly decreased, likely due to **overfitting** or **GPU limitations**:
     - **Mean Accuracy All Class Average**: 65.75%
     - **Mean Accuracy Images**: 76.92%
     - **Mean Precision**: 79.24%
     - **Mean Recall**: 78.02%
     - **Mean F1 Score**: 74.83%

7. **Challenges Faced**:
   - **Overfitting**: Suspected due to continuing training after 10 epochs with a pre-trained model.
   - **GPU Limitations**: Training large models (300 million parameters) in one epoch due to memory and compute restrictions of the GPU (6GB VRAM).

8. **Next Steps**:
   - Plan to fine-tune the pre-trained model while avoiding overfitting by adjusting the learning rate, epochs, and potentially applying **early stopping**.
   - Exploring data augmentation strategies for improving generalization and model performance.

Let me know if you'd like any additional details or modifications to this summary!
