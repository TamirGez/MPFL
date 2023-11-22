import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image
import json
import os


class CustomDataset(Dataset):
    def __init__(self, base_dataset: Dataset, num_classes: int, transforms: Compose = None):
        self.base_dataset = base_dataset
        # other initializations
        self.transforms = transforms

        # Number of classes
        self.num_classes = num_classes

        # By default, label_map is in regular order
        self.label_map = list(range(num_classes))

        # By default, do not add random noise
        self.use_random = False

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        :return: The number of items in the dataset.
        """
        return len(self.base_dataset)

    def shuffle_labels(self):
        self.label_map = torch.randperm(len(self.label_map)).tolist()

    def enable_random_noise(self):
        self.use_random = True

    def disable_random_noise(self):
        self.use_random = False

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        label = self.label_map[label]

        if self.use_random:
            # Flatten the image to a single dimension while keeping the channel order
            c, h, w = img.shape
            flattened = img.view(c, -1)  # shape becomes (C, H*W)
            # Shuffle each channel independently
            torch.manual_seed(index)
            torch.cuda.manual_seed_all(index)
            for i in range(c):
                flattened[i] = flattened[i][torch.randperm(h * w)]
            img = flattened.view(c, h, w)

        return img, label


class ImageNet100Dataset(Dataset):
    # data downloaded from https://www.kaggle.com/datasets/ambityga/imagenet100/code
    def __init__(self, root_dir, label_file, transform=None):
        self.transforms = transform
        self.labels, self.base_dataset = self._load_dataset(root_dir, label_file)
        self.num_classes = len(self.labels)
        self.label_to_index = {v: k for k, v in enumerate(self.labels.keys())}

    @staticmethod
    def _load_dataset(root_dir, label_file):
        images = []
        labels = json.load(open(label_file))

        for label_dir in os.listdir(root_dir):
            label_dir_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_dir_path):
                for img_file in os.listdir(label_dir_path):
                    if img_file.lower().endswith('jpeg'):
                        img_file_path = os.path.join(label_dir_path, img_file)
                        images.append((img_file_path, label_dir))
        return labels, images

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img_path, label_dir = self.base_dataset[index]
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        label_index = self.label_to_index[label_dir]
        return image, label_index


