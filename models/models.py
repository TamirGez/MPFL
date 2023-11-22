import os
import torch
import torch.nn.utils.prune as prune
import torchvision
from abc import ABC
from copy import deepcopy
from typing import Dict, Optional
import logging
from tqdm import tqdm
# local imports
from utils.model_utils import select_loss_function, select_optimizer


class ModelClass(ABC):
    def __init__(self, mask: Dict, train_loader, val_loader, test_loader, num_classes: int,
                 device: str = "cpu", logger: Optional[logging.Logger] = None):
        """
        Initialize the ModelClass object.

        :param mask: A dict mask for each layer to set for pruning.
        :param train_loader: A PyTorch DataLoader for loading training data.
        :param val_loader: A PyTorch DataLoader for loading validation data.
        :param test_loader: A PyTorch DataLoader for loading test data.
        :param num_classes: Total number of classes in the data.
        :param device: Device to run the model on. Default is "cpu".
        :param logger: Logger for logging information. Default is None.
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.mask = mask
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.num_classes = num_classes

    def create_train_env(self, args, mask=None):
        """
        Create the training environment including loss function, optimizer, and learning rate scheduler.

        :param mask: pruning mask for the model
        :param args: Arguments including optimizer, learning rate, and other parameters.
        """
        # define model
        mask = mask if mask is not None else {}
        self.cleanup()
        self.model = self.create_model(self.num_classes)
        self.create_mask(0.0)
        self.set_mask(mask)
        # Define loss function, optimizer and learning rate scheduler
        self.criterion = select_loss_function(args.loss_function, gamma=args.gamma_loss, alpha=args.alpha,
                                              smoothing=args.smoothing, neg_pos_ratio=args.neg_pos_ratio)
        self.optimizer = select_optimizer(args.optimizer, self.model, lr=args.learning_rate, momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.step_size],
                                                              gamma=args.gamma)

    def create_model(self, num_classes=0) -> torch.nn.Module:
        """
        To be implemented in subclass. This should return the model to be used.

        :param num_classes: Total number of classes.
        :return: Model to be used.
        """
        raise NotImplementedError

    def get_weights(self):
        """
        Get the current weights of the model.

        :return: A state dictionary containing the weights of the model.
        """
        if self.model:
            return self.model.state_dict()
        else:
            self.logger.error("No model is defined to get weights from.")
            return None

    def set_weights(self, state_dict):
        """
        Set the weights of the model to the provided state dictionary.

        :param state_dict: A state dictionary containing the weights to be loaded into the model.
        """
        if self.model:
            self.model.load_state_dict(state_dict)
            self.logger.info("Model weights updated successfully.")
        else:
            self.logger.error("No model is defined to set weights on.")

    def load_model(self, save_path: str):
        """
        Load the model state from the specified path.

        :param save_path: Path to load the model state.
        :type save_path: str
        :raises AssertionError: If the model does not exist at the specified path.
        """
        model_path = os.path.join(save_path, 'best_model.pt')
        try:
            assert os.path.exists(model_path), "Model does not exist at specified path"
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"Model loaded successfully from {model_path}")
        except AssertionError as error:
            self.logger.error(error)

    def set_mask(self, mask: Dict):
        """
        Set the mask for pruning.

        :param mask: Dictionary containing masks for each layer.
        """
        if mask is None:
            return
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and name in mask:
                with torch.no_grad():
                    module.weight.data = mask[name] * module.weight_orig
                    module.weight_mask = deepcopy(mask[name])

    def create_mask(self, prune_factor: float = 0.0) -> Dict:
        """
        Create a mask for pruning.

        :param prune_factor: Factor for pruning.
        :return: Dictionary containing masks for each layer.
        """
        mask = {}
        self.model.to(self.device)
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                prune.ln_structured(m, name="weight", amount=prune_factor, n=2, dim=0)
                mask[name] = (m.weight != 0).int().to('cpu')
        self.model.to("cpu")
        return mask

    def train(self, epochs=100, print_interval=2000, save_path=None, accuracy=True, print_results=True):
        """
        Trains a given model using a specified optimizer and loss function, with optional learning rate scheduling.
        :param epochs: The number of epochs to train for (default: 100).
        :param print_interval: The number of batches between print statements (default: 2000).
        :param save_path: The path to save the trained model (default: current working directory).
        :param accuracy: If True, calculates and prints accuracy on validation set, otherwise prints validation loss (default: True).
        :param print_results: If True, print training process and train results (default: True).
        """
        model_path = os.path.join(save_path, 'best_model.pt')
        # removing old runs model.
        if os.path.isfile(model_path):
            os.remove(model_path)
            self.logger.debug(f"Old model removed from {model_path}")

        for epoch in range(epochs):  # loop over the specified number of epochs
            self.logger.info(f"Epoch[{epoch + 1} / {epochs}]")
            self._train_one_epoch(accuracy, print_interval, print_results)
            self._validate_one_epoch(save_path, accuracy, print_results)
        if print_results:
            self.logger.info('Finished Training')
        if accuracy:
            return self.best_accuracy
        else:
            return self.best_loss

    def _train_one_epoch(self, accuracy, print_interval, print_results):
        running_loss = 0.0
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0
        self.model.train()
        for i, (inputs, labels) in tqdm(enumerate(self.train_loader), 'training', total=len(self.train_loader)):
            # convert data to device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # zero the gradients
            self.optimizer.zero_grad()
            # forward pass
            outputs = self.model(inputs)
            # calculate loss
            loss = self.criterion(outputs, labels)
            # backward pass
            loss.backward()
            # update model parameters
            self.optimizer.step()
            # accumulate the loss
            running_loss += loss.item()
            # calculate accuracy on training batch
            if accuracy:
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            if i % print_interval == print_interval - 1 and print_results:  # print the loss at specified intervals
                self.logger.info(
                    f"Batch [{i + 1}/{len(self.train_loader)}], Training Loss: {running_loss / print_interval:.3f}")
                epoch_loss += running_loss
                running_loss = 0.0
        self.optimizer.zero_grad()
        # adjust learning rate based on the scheduler
        self.scheduler.step()
        epoch_loss += running_loss
        epoch_loss /= len(self.train_loader)  # calculate average epoch loss
        if print_results:
            self.logger.info(f"Training Loss: {epoch_loss:.3f}")

    def _validate_one_epoch(self, save_path, accuracy, print_results):
        # Validate the model on the validation set
        val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        # disable gradient computation to save memory
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, 'validating', total=len(self.val_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, labels).item()

                # calculate accuracy on validation batch
                if accuracy:
                    _, predicted = torch.max(outputs, dim=1)
                    total_val_samples += len(predicted)
                    total_val_correct += (predicted == labels).sum().item()

            val_loss /= len(self.val_loader)  # calculate average validation loss

            # Print validation loss or accuracy
            if accuracy:
                val_accuracy = 100.0 * total_val_correct / total_val_samples
                if print_results:
                    self.logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            else:
                if print_results:
                    self.logger.info(f"Validation Loss: {val_loss:.3f}")
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pt'))
        self.logger.info(f"Acc: {100 * total_val_correct / total_val_samples: .3f} % ")

    def evaluate_model(self, save_path=os.getcwd()):
        """
        Evaluates a trained model on a test dataset, and prints the accuracy of the best model.
        :param save_path: The path where the trained model is saved (default: current working directory).
        """
        # Load the best model saved during training
        self.load_model(save_path)
        # Initialize counters for correct predictions and total examples
        correct = 0
        total = 0
        # Disable gradient calculation for efficiency
        self.model.eval()
        with torch.no_grad():
            # Iterate over the test data loader
            for inputs, labels in self.test_loader:
                # Move inputs and labels to the appropriate device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Forward pass through the model
                outputs = self.model(inputs)
                # Get the predicted class by taking the index of the maximum log-probability
                _, predicted = torch.max(outputs.data, 1)
                # Update the total number of examples
                total += labels.size(0)
                # Update the number of correct predictions
                correct += (predicted == labels).sum().item()
        # Print the accuracy of the best model on the test set
        return self.model, 100 * correct / total

    def run_model(self, args, mask=None, state_dict=None):
        if mask is None:
            mask = self.mask
        self.create_train_env(args, mask)
        if state_dict is not None:
            self.set_weights(state_dict)
        self.model.to(self.device)
        self.train(epochs=args.epochs, print_interval=args.print_interval, save_path=args.save_path,
                   print_results=args.print_results)
        # Evaluate model
        self.model, accuracy = self.evaluate_model(save_path=args.save_path)
        if args.print_results:
            self.logger.info(f"Accuracy of the best model on the test set: {accuracy:.2f}%")
        # creating new mask
        self.model.to("cpu")
        return accuracy

    def set_new_mask(self, mask):
        self.mask = mask

    def cleanup(self):
        """
        Clean up resources used by the model.

        """
        del self.model
        del self.optimizer
        del self.criterion
        del self.scheduler
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()


class VGG11Model(ModelClass):
    def create_model(self, num_classes: int = 0) -> torch.nn.Module:
        """
        Create and return a VGG11 model, modifying the final layer to match the number of classes.

        :param num_classes: Total number of classes.
        :type num_classes: int, optional
        :return: The modified VGG11 model.
        :rtype: torch.nn.Module
        """
        self.logger.info(f"Creating VGG11 model with {num_classes} classes.")
        model = torchvision.models.vgg11(pretrained=True)
        if num_classes:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
        return model


class ResNet34Model(ModelClass):
    def create_model(self, num_classes: int = 0) -> torch.nn.Module:
        """
        Create and return a ResNet34 model, modifying the final layer to match the number of classes.

        :param num_classes: Total number of classes.
        :type num_classes: int, optional
        :return: The modified ResNet34 model.
        :rtype: torch.nn.Module
        """
        self.logger.info(f"Creating ResNet34 model with {num_classes} classes.")
        model = torchvision.models.resnet34(pretrained=True)
        if num_classes:
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        return model


class ResNet18Model(ModelClass):
    def create_model(self, num_classes: int = 0) -> torch.nn.Module:
        """
        Create and return a ResNet34 model, modifying the final layer to match the number of classes.

        :param num_classes: Total number of classes.
        :type num_classes: int, optional
        :return: The modified ResNet34 model.
        :rtype: torch.nn.Module
        """
        self.logger.info(f"Creating ResNet18 model with {num_classes} classes.")
        model = torchvision.models.resnet18(pretrained=True)
        if num_classes:
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        return model


def pick_model(model_name: str):
    loaders = {
        'vgg11': VGG11Model,
        'resnet34': ResNet34Model,
        'resnet18': ResNet18Model
    }

    model_name = model_name.lower()
    if model_name not in loaders:
        raise ValueError(f"Invalid model name '{model_name}'.")

    return loaders[model_name]
