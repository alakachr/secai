import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(
        self,
        image_folder="/data/ubuntu/secai/data/Prestamo_no_consensuado.v5i.coco/train",
        transform=None,
        json_file="/data/ubuntu/secai/data/Prestamo_no_consensuado.v5i.coco/train/_annotations.coco.json",
    ):
        self.json_file = json_file
        self.image_folder = image_folder
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        with open(self.json_file, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image_info = self.data["images"][idx]
        image_id = image_info["id"]
        image_path = os.path.join(self.image_folder, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []

        for annotation in self.data["annotations"]:
            if annotation["image_id"] == image_id:
                x, y, width, height = annotation["bbox"]
                x_min, y_min, x_max, y_max = x, y, x + width, y + height
                category_id = annotation["category_id"]
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(category_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
