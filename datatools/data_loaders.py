import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import numpy as np
import torchvision
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import logging
# local imports
from datatools.custom_datasets import ImageNet100Dataset, CustomDataset


class DataLoaderBase(ABC):
    """
    An abstract base class for a DataLoader.

    :param root: The root directory where the dataset is stored.
    :param batch_size: The batch size for the DataLoader.
    :param num_workers: The number of worker threads to use for loading the data.
    :param val_ratio: The ratio of the dataset to use for validation.
    :param edges_number: The number of edges in the dataset.
    :param logger: A logging object to log messages.
    """

    def __init__(self, root: str = './data', batch_size: int = 64, num_workers: int = 4, val_ratio: float = 0.2,
                 edges_number: int = 10, logger: logging.Logger = None):
        """
        Initialize a new DataLoaderBase instance.

        :param root: The root directory where the dataset is stored.
        :param batch_size: The batch size for the DataLoader.
        :param num_workers: The number of worker threads for loading the data.
        :param val_ratio: The ratio of the dataset to use for validation.
        :param edges_number: The number of edges in the dataset.
        :param logger: A logging object to log messages.
        """
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.edges_number = edges_number
        self.num_classes = 0
        self.train_dataset, self.test_dataset = self.load_data()

    @abstractmethod
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Abstract method to load dataset. Must be implemented by subclasses.

        :return: A tuple of train and test datasets.
        """
        pass

    def create_indices(self) -> Tuple[List[int], List[int]]:
        """
        Creates indices for training and validation datasets.

        :return: A tuple containing lists of training and validation indices.
        """
        self.logger.debug("Creating indices...")
        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.val_ratio * num_train))
        np.random.shuffle(indices)
        return indices[split:], indices[:split]

    def shuffle_labels(self, worker_loader: Dict[str, CustomDataset]):
        """
        Shuffles the labels of a specific worker loader.

        :param worker_loader: A dictionary containing DataLoader objects for a worker.
        """
        self.logger.debug("Shuffle the labels...")
        worker_loader['train'].shuffle_labels()
        worker_loader['val'].shuffle_labels()
        worker_loader['val'].label_map = worker_loader['train'].label_map

    def noise_input(self, worker_loader: Dict[str, CustomDataset]):
        """
        Changes the input from the image into a random noise unit.

        :param worker_loader: A dictionary containing DataLoader objects for a worker.
        """
        self.logger.debug("Adding noise to input...")
        worker_loader['train'].enable_random_noise()
        worker_loader['val'].enable_random_noise()

    def convert_to_dataloader(self, train_subset, val_subset, test_dataset) -> Dict[str, DataLoader]:
        self.logger.debug("Converting to DataLoader objects...")
        dataloader = {
            'train': DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, collate_fn=self.collate_fn),
            'val': DataLoader(val_subset, batch_size=2 * self.batch_size, shuffle=False,
                              num_workers=self.num_workers, collate_fn=self.collate_fn),
            'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                               num_workers=self.num_workers, collate_fn=self.collate_fn)
        }
        return dataloader

    @staticmethod
    def collate_fn(batch):
        data, target = zip(*batch)
        data = torch.stack(data)
        target = torch.tensor(target)
        return data, target

    def create_loaders(self, train_idx: List[int], val_idx: List[int]) -> List[Dict[str, CustomDataset]]:
        self.logger.debug("Creating data loaders...")
        workers_loaders = []
        train_batch = int(len(train_idx) / self.edges_number)
        val_batch = int(len(val_idx) / self.edges_number)

        for i in range(self.edges_number):
            train_start = i * train_batch
            train_end = (i + 1) * train_batch if i != self.edges_number - 1 else len(train_idx)
            val_start = i * val_batch
            val_end = (i + 1) * val_batch if i != self.edges_number - 1 else len(val_idx)

            # Sliced train_indices and val_indices
            train_indices = train_idx[train_start:train_end]
            val_indices = val_idx[val_start:val_end]

            # Create new CustomDataset instances for each worker
            worker_train_dataset = CustomDataset(Subset(self.train_dataset, train_indices), self.num_classes)
            worker_val_dataset = CustomDataset(Subset(self.train_dataset, val_indices), self.num_classes)

            # Create data loaders for each worker
            workers_loaders.append({'train': worker_train_dataset, 'val': worker_val_dataset})

        return workers_loaders

    def create_global_loaders(self, worker_loader: List[Dict[str, Subset]]) -> Dict[str, DataLoader]:
        """
        Creates global DataLoader objects by concatenating the datasets from worker loaders.

        :param worker_loader: A list of dictionaries containing Subset objects.
        :return: A dictionary of global DataLoader objects.
        """
        self.logger.debug("Creating global data loaders...")
        # Concatenating the training and validation datasets from each worker
        global_train_dataset = ConcatDataset([worker['train'] for worker in worker_loader])
        global_val_dataset = ConcatDataset([worker['val'] for worker in worker_loader])

        # Creating global DataLoader objects
        loaders = self.convert_to_dataloader(global_train_dataset, global_val_dataset, self.test_dataset)

        return loaders

    def create_as_loaders(self, worker_loader: List[Dict[str, Subset]]):
        self.logger.debug("Creating asynchronous data loaders...")
        worker_dataloader = [self.convert_to_dataloader(worker['train'], worker['val'], self.test_dataset) for worker in
                             worker_loader]
        global_loader = self.create_global_loaders(worker_loader)
        return worker_dataloader, global_loader


class CIFAR10DataLoader(DataLoaderBase):
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the CIFAR10 dataset.

        :return: A tuple of train and test datasets.
        """
        self.logger.debug("Loading CIFAR10 dataset")
        try:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_dataset = datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform_test)

            # Setting num_classes here
            self.num_classes = len(train_dataset.classes)
            return train_dataset, test_dataset

        except Exception as e:
            self.logger.error(f"Error loading CIFAR10 dataset: {e}")  # Log the error message
            raise e  # Re-raise the exception to handle it upstream


class CIFAR100DataLoader(DataLoaderBase):
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the CIFAR100 dataset.

        :return: A tuple of train and test datasets.
        """
        self.logger.debug("Loading CIFAR100 dataset")
        # Define transformations, similar to your get_cifar100_data function
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        # Load the CIFAR100 dataset
        train_dataset = torchvision.datasets.CIFAR100(root=self.root, train=True, download=True,
                                                      transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=self.root, train=False, download=True,
                                                     transform=transform_test)

        # Setting num_classes here
        self.num_classes = len(train_dataset.classes)
        return train_dataset, test_dataset


class ImageNet100Loader(DataLoaderBase):
    def load_data(self) -> Tuple:
        """
        Loads the ImageNet100 dataset.

        :return: A tuple of train and test datasets.
        """
        self.logger.debug("Loading ImageNet100 dataset")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        data_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/imagenet100')
        label_file = os.path.join(data_root_dir, 'Labels.json')
        train_dataset = ImageNet100Dataset(root_dir=os.path.join(data_root_dir, 'train'), label_file=label_file,
                                           transform=transform)
        test_dataset = ImageNet100Dataset(root_dir=os.path.join(data_root_dir, 'val'), label_file=label_file,
                                          transform=transform)
        self.num_classes = len(train_dataset.labels)
        return train_dataset, test_dataset


def dataloader_factory(data_type: str, **kwargs) -> DataLoaderBase:
    loaders = {
        'cifar10': CIFAR10DataLoader,
        'cifar100': CIFAR100DataLoader,
        'imagenet100': ImageNet100Loader
    }

    data_type = data_type.lower()
    if data_type not in loaders:
        raise ValueError(f"Invalid dataset name '{data_type}'.")

    return loaders[data_type](**kwargs)


def create_loaders(data_type: str, logger: logging.Logger = None, add_random_unit: bool = False,
                   add_shuffle_unit: bool = False, batch_size: int = 20, num_workers: int = 1, val_ratio: float = 0.2,
                   edges_number: int = 10):
    logger = logging.getLogger(__name__) if logger is None else logger

    logger.info(f'loading {data_type} dataset')

    data_object_args = {
        'root': './data',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'val_ratio': val_ratio,
        'edges_number': edges_number
    }

    data_object = dataloader_factory(data_type, **data_object_args)

    train_indices, val_indices = data_object.create_indices()
    workers_loaders = data_object.create_loaders(train_indices, val_indices)

    if add_random_unit:
        data_object.noise_input(workers_loaders[0])
    if add_shuffle_unit:
        data_object.shuffle_labels(workers_loaders[1])

    worker_dataloader, global_loader = data_object.create_as_loaders(workers_loaders)

    return worker_dataloader, global_loader, data_object.num_classes


if __name__ == "__main__":
    import pylab as plt

    worker_dataloader, global_loader, num_classes = create_loaders('imagenet100', add_random_unit=False,
                                                                   add_shuffle_unit=False, batch_size=16)
    for batch in worker_dataloader[0]['train']:
        print(f"got a batch out of {len(worker_dataloader[0]['train'])}")
        break
    print('done')
