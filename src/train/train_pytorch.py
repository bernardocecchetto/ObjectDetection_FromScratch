import sys

sys.path.append(".")

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import cv2 as cv2
import pandas as pd
import wandb
from PIL import Image
import albumentations as A

# torchvision libraries
import torch
import torchvision


from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)
import albumentations as A
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from albumentations.pytorch.transforms import ToTensorV2

from src.pytorch_functions import train_one_epoch, evaluate, collate_fn

torch.cuda.empty_cache()


class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, type, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.type = type
        self.width = width
        self.height = height

        # sorting the images for consistency
        # annotation file
        self.df_annot = pd.read_csv(
            f"F:/ObjectDetection_FromScratch/data/annotations/{self.type}_annotations.csv"
        )

        # To get images, the extension of the filename is checked to be jpg
        # check if any image is not in the csv. If not, exclude it

        self.imgs = [
            image
            for image in sorted(os.listdir(files_dir))
            if image in self.df_annot["filename"].values
        ]

        # classes: 0 index is reserved for background
        self.classes = [
            "person",
            "horse",
            "aeroplane",
            "tvmonitor",
            "motorbike",
            "dog",
            "sofa",
            "bicycle",
            "car",
            "diningtable",
            "train",
            "bus",
            "cow",
            "chair",
            "cat",
            "bird",
            "pottedplant",
            "boat",
            "sheep",
            "bottle",
        ]

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        df_img = self.df_annot[self.df_annot.filename == img_name]

        boxes = []
        labels = []
        num_objs = len(df_img)
        # mask_list = []
        for idx, row in df_img.iterrows():
            # cv2 image gives size as height x width
            wt = img_rgb.shape[1]
            ht = img_rgb.shape[0]

            # box coordinates for xml files are extracted and corrected for image size given
            labels.append(self.classes.index(row["class"]))

            # bounding box
            xmin = int(row["xmin"])
            xmax = int(row["xmax"])
            ymin = int(row["ymin"])
            ymax = int(row["ymax"])

            # new coordinates after resizing
            xmin = int((xmin / wt) * self.width)
            xmax = int((xmax / wt) * self.width)
            ymin = int((ymin / ht) * self.height)
            ymax = int((ymax / ht) * self.height)

            # creating the mask
            # mask = np.zeros(img_array.shape[:2], dtype="uint8")
            # cv2.rectangle(mask, (xmin, xmax), (ymin, ymax), 255, -1)
            # # masked = cv2.bitwise_and(img_array, img_array, mask=mask)
            # mask_list.append(mask)
            boxes.append([xmin, ymin, xmax, ymax])
            # display the mask, and the output image
            # cv2.imshow("Mask", mask)
            # cv2.waitKey(0)
            # cv2.imshow("Masked Image", masked)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # mask_array = np.array(mask_list)
        # masks = torch.as_tensor(mask_array, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        # suppose all instances are not crowd
        iscrowd = torch.zeros(num_objs, dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            sample = self.transforms(
                image=img_res, bboxes=target["boxes"], labels=labels
            )

            img_res = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return img_res, target

    def __len__(self):
        return len(self.imgs)


def get_object_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(
    #     in_features_mask, hidden_layer, num_classes
    # )

    return model


def get_transform(train):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(0.5),
                # ToTensorV2 converts image to pytorch tensor without div by 255
                ToTensorV2(p=1.0),
            ],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
    else:
        return A.Compose(
            [ToTensorV2(p=1.0)],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )


# def get_transform(train):
#     transforms = []
#     transforms.append(PILToTensor())
#     transforms.append(ConvertImageDtype(torch.float))
#     if train:
#         transforms.append(RandomHorizontalFlip(0.5))
#     return Compose(transforms)


def main():
    # defining the files directory and testing directory
    files_dir = (
        "F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow/train"
    )
    valid_dir = (
        "F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow/valid"
    )

    # use our dataset and defined transformss
    dataset = PascalVOCDataset(
        files_dir, "train", width=448, height=448, transforms=get_transform(train=False)
    )
    dataset_valid = PascalVOCDataset(
        valid_dir, "valid", width=448, height=448, transforms=get_transform(train=False)
    )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, num_workers=10, collate_fn=collate_fn
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=5,
        shuffle=False,
        num_workers=10,
        collate_fn=collate_fn,
    )

    # to train on gpu if selected.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 20

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # training for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_valid, device=device)


if __name__ == "__main__":
    main()