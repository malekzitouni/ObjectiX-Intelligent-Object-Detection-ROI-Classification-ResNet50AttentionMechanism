import numpy as np
import os
import random
import cv2
from pycocotools.coco import COCO

class COCOReader:
    def __init__(self, ImageDir=r'C:\Users\Pc\Desktop\COCO_Dataset\train2017\train2017',
                 AnnotationFile=r'C:\Users\Pc\Desktop\COCO_Dataset\annotations_trainval2017\annotations\instances_train2017.json',
                 MaxBatchSize=100, MinSize=160, MaxSize=800, MaxPixels=800*800*5):
        self.ImageDir = ImageDir  # Image directory
        self.AnnotationFile = AnnotationFile  # File containing image annotation
        self.MaxBatchSize = MaxBatchSize  # Maximum number of images to be included in a single batch
        self.MinSize = MinSize  # Minimum image width and height included in a single batch
        self.MaxSize = MaxSize  # Maximum image width and height included in a single batch
        self.MaxPixels = MaxPixels  # Maximum number of pixels in all the batch (reduce to solve out of memory issues)

        self.coco = COCO(AnnotationFile)  # Load annotation file
        self.cats = self.coco.loadCats(self.coco.getCatIds())  # List of categories
        self.NumCats = len(self.cats)  # Number of categories
        self.CatNames = [cat['name'] for cat in self.cats]  # Category names
        self.ImgIds = {i: self.coco.getImgIds(catIds=self.cats[i]['id']) for i in range(self.NumCats)}  # List of ids of images containing various categories
        self.ClassItr = np.zeros(self.NumCats)  # Class iterator to track category progress

    def ReadNextBatchRandom(self):
        Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch height
        Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch width
        BatchSize = int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))  # Number of images in batch

        BImgs = np.zeros((BatchSize, Hb, Wb, 3), dtype=np.float32)  # Images
        BSegmentMask = np.zeros((BatchSize, Hb, Wb), dtype=np.float32)  # Segment mask
        BLabels = np.zeros((BatchSize), dtype=int)  # Class labels
        BLabelsOneHot = np.zeros((BatchSize, self.NumCats), dtype=np.float32)  # One-hot encoding for classes

        for i in range(BatchSize):
            ClassNum = np.random.randint(self.NumCats)  # Choose random category
            ImgNum = np.random.randint(len(self.ImgIds[ClassNum]))  # Choose random image
            ImgData = self.coco.loadImgs(self.ImgIds[ClassNum][ImgNum])[0]  # Pick image data
            image_name = ImgData['file_name']  # Get image name
            Img = cv2.imread(os.path.join(self.ImageDir, image_name))  # Load image

            if Img is None:
                print(f"Warning: Image {image_name} could not be loaded.")
                continue

            if Img.ndim == 2:  # If grayscale, convert to RGB
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)

            Img = Img[:, :, :3]  # Get first 3 channels in case there are more
            annIds = self.coco.getAnnIds(imgIds=ImgData['id'], catIds=self.cats[ClassNum]['id'], iscrowd=None)
            InsAnn = self.coco.loadAnns(annIds)  # Array of instance annotations

            if len(InsAnn) == 0:
                print(f"No annotations found for image {image_name}.")
                continue

            Ins = InsAnn[np.random.randint(len(InsAnn))]  # Choose random instance
            Mask = self.coco.annToMask(Ins)  # Get mask (binary map)
            bbox = np.array(Ins['bbox']).astype(np.float32)  # Get instance bounding box

            h, w, d = Img.shape
            Rs = max(Hb / h, Wb / w)  # Resize factor

            if Rs > 1:  # Resize image and mask if necessary
                h = int(max(h * Rs, Hb))
                w = int(max(w * Rs, Wb))
                Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                Mask = cv2.resize(Mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
                bbox *= Rs

            x1, y1 = int(bbox[0]), int(bbox[1])
            Wbox, Hbox = int(bbox[2]), int(bbox[3])  # Bounding box dimensions

            Xmin = max(0, x1 - (Wb - Wbox) if Wb > Wbox else x1)
            Xmax = min(w - Wb, x1 if Wb > Wbox else x1 + (Wbox - Wb))

            Ymin = max(0, y1 - (Hb - Hbox) if Hb > Hbox else y1)
            Ymax = min(h - Hb, y1 if Hb > Hbox else y1 + (Hbox - Hb))

            if Xmax < Xmin or Ymax < Ymin:
                print(f"Invalid bounding box for image {image_name}. Xmin: {Xmin}, Xmax: {Xmax}, Ymin: {Ymin}, Ymax: {Ymax}")
                continue

            x0 = np.random.randint(low=Xmin, high=Xmax + 1)
            y0 = np.random.randint(low=Ymin, high=Ymax + 1)

            # Crop images and masks
            Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
            Mask = Mask[y0:y0 + Hb, x0:x0 + Wb]

            if random.random() < 0.0:  # Augment the image by mirroring (change probability as needed)
                Img = np.fliplr(Img)
                Mask = np.fliplr(Mask)

            BImgs[i] = Img
            BSegmentMask[i, :, :] = Mask
            BLabels[i] = ClassNum
            BLabelsOneHot[i, ClassNum] = 1

        return BImgs, BSegmentMask, BLabels, BLabelsOneHot

    def ReadSingleImageAndClass(self, ClassNum, ImgNum):
        ImgData = self.coco.loadImgs(self.ImgIds[ClassNum][ImgNum])[0]  # Pick image data
        image_name = ImgData['file_name']  # Get image name
        Img = cv2.imread(os.path.join(self.ImageDir, image_name))  # Load image

        if Img is None:
            print(f"Warning: Image {image_name} could not be loaded.")
            return None, None, None, None

        if Img.ndim == 2:  # If grayscale, convert to RGB
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)

        Img = Img[:, :, :3]  # Get first 3 channels
        annIds = self.coco.getAnnIds(imgIds=ImgData['id'], catIds=self.cats[ClassNum]['id'], iscrowd=None)  # Get list of annotation for image (of the specific class)
        InsAnn = self.coco.loadAnns(annIds)  # Create array of instance annotation

        if len(InsAnn) == 0:
            print(f"No annotations found for image {image_name}.")
            return None, None, None, None

        [Hb, Wb, d] = Img.shape
        BatchSize = len(InsAnn)
        BImgs = np.zeros((BatchSize, Hb, Wb, 3), dtype=np.float32)  # Images
        BSegmentMask = np.zeros((BatchSize, Hb, Wb), dtype=np.float32)  # Segment mask
        BLabels = np.zeros((BatchSize), dtype=int)  # Class of batch
        BLabelsOneHot = np.zeros((BatchSize, self.NumCats), dtype=np.float32)  # Batch classes in one hot

        for i in range(BatchSize):
            BImgs[i] = Img
            BSegmentMask[i, :, :] = self.coco.annToMask(InsAnn[i])  # Get mask (binary map)
            BLabels[i] = ClassNum
            BLabelsOneHot[i, ClassNum] = 1  # Set one-hot encoding label

        return BImgs, BSegmentMask, BLabels, BLabelsOneHot
