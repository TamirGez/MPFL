import random
import numpy as np
import torch
import pylab as plt
import logging
from logging.handlers import RotatingFileHandler
import os
from collections import OrderedDict
import json
import argparse


def set_seed(seed: int) -> None:
    """Sets the random seed values for reproducibility.

    :param seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(output_dir: str, module_name: str):
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging to output to a file, with a rotating file handler
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create a logger object
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers if this function is called multiple times
    if not logger.handlers:
        # Create a file handler that logs debug and higher level messages
        log_file = os.path.join(output_dir, 'log.log')
        file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5,
                                           backupCount=5)  # 5 MB per file, max 5 files
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the file handler
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)

        # If you also want to log to the console, add a StreamHandler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def average_weights(edge_weights):
    """
    Averages the weights from different model state dictionaries.

    :param edge_weights: A list of state dictionaries from the edge models.
    :return: A state dictionary with the averaged weights.
    """
    # Initialize a new state dictionary with the same keys and empty lists to hold weights
    average_dict = OrderedDict({k: torch.zeros_like(v, dtype=torch.float) for k, v in edge_weights[0].items()})

    # Sum all weights
    for weight in edge_weights:
        for key in average_dict.keys():
            # Ensure the weights are float before summing
            average_dict[key] += weight[key].float()

    # Divide by the number of edge models to get the average
    for key in average_dict.keys():
        average_dict[key] /= len(edge_weights)

    return average_dict


def plot_results(pruning_voting_sparsity, federated_sparsity, center_sparsity, plot_title, output_dir):
    plt.figure()
    plt.plot(list(pruning_voting_sparsity.values()), list(pruning_voting_sparsity.keys()), '-*')
    plt.plot(list(federated_sparsity.values()), list(federated_sparsity.keys()), '--o')
    plt.plot(list(center_sparsity.values()), list(center_sparsity.keys()), '-.')
    plt.xlabel('Sparsity (%)')
    plt.ylabel('Accuracy (%)')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend(['Our Algorithm', 'Federated Algorithm', 'Center Model'])
    plt.savefig(os.path.join(output_dir, plot_title))
    plt.close()

    # Save the sparsity-accuracy results as JSON
    sparsity_accuracy_results = {
        'pruning_voting_sparsity': pruning_voting_sparsity,
        'federated_sparsity': federated_sparsity,
        'center_sparsity': center_sparsity
    }
    with open(os.path.join(output_dir, 'sparsity_accuracy_results.json'), 'w') as f:
        json.dump(sparsity_accuracy_results, f, indent=4)


def prase_input():
    # Define argparse
    parser = argparse.ArgumentParser(description='Pruning Voting')

    # Randomization & Reporting args
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--print_results', type=bool, default=True,
                        help='Printing training process and results (default: True)')
    parser.add_argument('--print_interval', type=int, default=2000,
                        help='Interval for printing the training loss (default: 2000)')

    # Model & Data related args
    parser.add_argument('--edges_number', type=int, default=10, help='Edge devices number (default: 10)')
    parser.add_argument('--model_name', type=str, default='vgg11', choices=['vgg11', 'resnet34', 'resnet18'],
                        help='Model type (default: vgg11)')
    parser.add_argument('--data_type', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'imagenet100'],
                        help='Data type (default: cifar100)')

    # Noise & Transformation related args
    parser.add_argument('--add_random_unit', type=bool, default=False, help='Add random noise to a unit')
    parser.add_argument('--add_shuffle_unit', type=bool, default=False, help='Shuffle a unit')

    # Pruning related args
    parser.add_argument('--prune_factor', type=float, default=0.1, help='Prune factor for each round (default: 0.1)')
    parser.add_argument('--use_histogram', type=bool, default=False,
                        help='Use histogram to calculate server mask (default: False)')
    parser.add_argument('--histogram_percentile', type=float, default=0.95,
                        help='Histogram percentile for server mask (default: 0.95)')
    parser.add_argument('--iterating_times', type=int, default=9,
                        help='Iterations between server and edges (default: 9)')

    # Training and Optimizer related args
    parser.add_argument('--plot_title', type=str, default='results.png', help='plot_title')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation data split ratio (default: 0.1)')
    parser.add_argument('--num_workers', type=int, default=4, help='input number of workers for training (default: 4)')
    parser.add_argument('--loss_function', type=int, default=0, help='loss function to use (default: 0)')
    parser.add_argument('--gamma_loss', type=float, default=2, help='The gamma value for FocalLoss (optional)')
    parser.add_argument('--alpha', type=float, default=0.25, help='The alpha value for FocalLoss or SSD loss(optional)')
    parser.add_argument('--neg_pos_ratio', type=float, default=0.25, help='The neg_pos_ratio for SSD loss (optional)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='The smoothing value for LabelSmoothingLoss (optional)')
    parser.add_argument('--optimizer', type=int, default=0, help='optimizer to use (default: 0)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay (default: 5e-5)')
    parser.add_argument('--nesterov', type=bool, default=True, help='using nesterov in optimizer (default: True)')
    parser.add_argument('--step_size', type=int, default=45, help='learning rate scheduler step size (default: 45)')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler gamma (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help='path to save\\load the model and results')

    args = parser.parse_args()
    return args


def model_params(model):
    total_size = 0
    mask_total_bits = 0
    for layer in model.state_dict().values():
        if len(layer.shape)==0:
            continue
        layer_size = 1
        for l in layer.shape:
            layer_size *= l
        total_size += layer_size
        mask_total_bits += layer.shape[0]
    return total_size, mask_total_bits
