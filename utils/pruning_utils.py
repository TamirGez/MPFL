import torch
import numpy as np
import os
import pylab as plt


def create_server_mask(edge_masks,
                       saving_path='',
                       required_prune=0.0,
                       create_plots=False,
                       use_histogram=False,
                       histogram_percentile=0.9,
                       logger=None):
    """
    Create a global pruning mask from the edge masks using voting and optionally a histogram method.

    :param edge_masks: List of masks from edge devices.
    :param required_prune: Minimum required pruning level.
    :param create_plots: Boolean, if True, creates and saves a plot of masks.
    :param use_histogram: Boolean, if True, uses histogram method for calculating global mask.
    :param histogram_percentile: Percentile for histogram threshold.
    :param logger: logger to print results over
    :return: A tuple containing the global mask and the prune value.
    """
    new_server_mask = {}
    sum_mask = {}

    # Calculate the sum for each key
    for key in edge_masks[0]:
        mask_values = [edge_mask[key] for edge_mask in edge_masks]
        sum_mask[key] = sum(mask_values) / len(edge_masks)

    current_prune, zeros_amount = 0, 0
    for key, mask_values in sum_mask.items():
        if use_histogram:
            # If using histogram, flatten all corresponding masks from edge_masks
            flattened_values = torch.cat([edge_mask[key].flatten() for edge_mask in edge_masks]).cpu().numpy()
            # Calculate the histogram threshold
            histogram_threshold = np.percentile(flattened_values, histogram_percentile * 100)
            # Update the new server mask based on the histogram threshold
            new_server_mask[key] = (mask_values >= histogram_threshold).to(edge_masks[0][key].dtype)
        else:
            for i in range(len(edge_masks)):
                # Calculate dynamic threshold
                dynamic_th = (len(edge_masks) - i) / len(edge_masks)
                # Update new server mask based on dynamic threshold
                new_server_mask[key] = (mask_values >= dynamic_th).to(edge_masks[0][key].dtype)
                new_zeros_amount = float(new_server_mask[key].nelement())
                new_pruned_num = float(torch.sum(new_server_mask[key] == 0))
                # Break the loop if pruning criterion is met
                if not new_zeros_amount or new_pruned_num / new_zeros_amount >= required_prune:
                    break

        # Calculate the number of zeros and total elements in the new server mask
        new_zeros_amount = float(new_server_mask[key].nelement())
        new_pruned_num = float(torch.sum(new_server_mask[key] == 0))
        current_prune += new_pruned_num
        zeros_amount += new_zeros_amount

    # Calculate the final pruning value
    prune_value = 100. * current_prune / zeros_amount

    if create_plots:
        # Create directory to save plots and masks
        saving_path = saving_path if len(saving_path) else os.getcwd()
        plots_saving_path = os.path.join(saving_path, "plots", str(prune_value))
        os.makedirs(plots_saving_path, exist_ok=True)
        full_flattened_values = []  # This will store all flattened values

        for key in sum_mask.keys():
            # Flatten the mask values and store them for the full histogram
            flattened_values = sum_mask[key].to('cpu').flatten().numpy()
            full_flattened_values.extend(flattened_values)  # Extend the list with flattened values

            # Plot and save histogram for each key
            vals, counts = np.unique(flattened_values, return_counts=True)
            plt.bar(vals, counts, color='blue')
            plt.title(f'Layer: {key}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plots_saving_path, f'{key}_histogram.png'))
            plt.close()

            # Save the mask
            torch.save(new_server_mask[key], os.path.join(plots_saving_path, f'{key}_mask.pth'))

        # Save the mask for all layers combined
        torch.save(new_server_mask, os.path.join(plots_saving_path, 'new_server_mask.pth'))

        # Plot and save the full histogram of all layers
        plt.hist(full_flattened_values, bins=50, color='green', alpha=0.75)
        plt.title('Histogram of All Layers Flattened')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(plots_saving_path, 'full_layers_histogram.png'))
        plt.close()

    if logger is not None:
        # Print information about the pruning
        logger.info(f"Final Pruning Value: {prune_value:.2f}%")
        logger.info(f"Total Zeros: {zeros_amount}")
        logger.info(f"Current Prune: {current_prune}")

    return new_server_mask, prune_value
