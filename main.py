import torch
from copy import deepcopy
import os
import json
import matplotlib

matplotlib.use('Agg')  # Use the non-GUI 'Agg' backend
# matplotlib.use('Qt5Agg')  # Switch to Qt5 backend, if available
# local imports
from utils.utils import set_seed, plot_results, create_logger, average_weights, prase_input
from datatools.data_loaders import create_loaders
from models.models import pick_model
from utils.pruning_utils import create_server_mask


def main():
    args = prase_input()
    output_dir = args.save_path

    # Setup logging
    logger = create_logger(output_dir, 'Simulation_Log')

    logger.info(f"Setting all random seeds to be {args.seed} for reproducibility.")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load data
    workers_loaders, loaders, num_classes = create_loaders(data_type=args.data_type,
                                                           logger=logger,
                                                           add_random_unit=args.add_random_unit,
                                                           add_shuffle_unit=args.add_shuffle_unit,
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           val_ratio=args.val_ratio,
                                                           edges_number=args.edges_number)
    logger.info(f"Loaded data with {args.data_type} type")

    model = pick_model(args.model_name)
    logger.info(f"Picked {args.model_name} model")

    # Save args as JSON in output dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)  # Convert args to dictionary before dumping

    logger.debug(f"Saved argparse to {os.path.join(output_dir, 'args.json')}")
    logger.debug(f"Command line arguments: {vars(args)}")  # Print the selected argparse in debug mode

    edge_units = [model(mask={}, train_loader=loader['train'], val_loader=loader['val'], test_loader=loader['test'],
                        num_classes=num_classes, device=device, logger=logger) for loader in workers_loaders]

    center_unit = model(mask={}, train_loader=loaders['train'], val_loader=loaders['val'], test_loader=loaders['test'],
                        num_classes=num_classes, device=device, logger=logger)
    center_unit.create_train_env(args)

    federated_sparsity = {}
    pruning_voting_sparsity = {}
    center_sparsity = {}
    federated_mask, pruning_voting_mask, center_mask = {}, {}, {}
    prune_size, federated_prune_size, center_prune_size = 0, 0, 0
    for itr in range(args.iterating_times + 1):
        logger.info(f'Starting run {itr + 1}/{args.iterating_times + 1}.')
        if prune_size > 90:
            break

        edge_masks = []
        edge_weights = []
        for i, edge_unit in enumerate(edge_units):
            logger.info(f'training edge unit {i + 1}/{len(edge_units)}')
            _ = edge_unit.run_model(args, mask=federated_mask)
            edge_weights.append(deepcopy(edge_unit.get_weights()))
            _ = edge_unit.run_model(args, mask=pruning_voting_mask)
            edge_masks.append(deepcopy(edge_unit.create_mask(args.prune_factor if itr else 0)))

        logger.info(f'Creating our algorithm mask.')
        pruning_voting_mask, prune_size = create_server_mask(edge_masks,
                                                             saving_path=output_dir,
                                                             required_prune=itr * args.prune_factor,
                                                             create_plots=args.print_results,
                                                             use_histogram=args.use_histogram,
                                                             histogram_percentile=args.histogram_percentile,
                                                             logger=logger)

        logger.info(f'Creating federated learning algorithm mask.')
        averaged_weights = average_weights(edge_weights)
        center_unit.set_weights(averaged_weights)
        federated_factor = 1 - (100 - prune_size) / (100 - federated_prune_size)
        federated_mask, federated_prune_size = create_server_mask([center_unit.create_mask(federated_factor)],
                                                                  required_prune=federated_factor, create_plots=False)

        logger.info(f'Creating center model algorithm mask.')
        logger.info('starting with training')
        _ = center_unit.run_model(args, mask=center_mask)

        logger.info('Calculating center model mask')
        center_factor = 1 - (100 - prune_size) / (100 - center_prune_size)
        center_mask, center_prune_size = create_server_mask([center_unit.create_mask(center_factor)],
                                                            required_prune=center_factor, create_plots=False)

        logger.info('Checking performance for comparison')
        pruning_voting_accuracy = center_unit.run_model(args, mask=pruning_voting_mask)
        federated_accuracy = center_unit.run_model(args, mask=federated_mask)
        center_accuracy = center_unit.run_model(args, mask=center_mask)

        federated_sparsity[federated_accuracy] = federated_prune_size
        pruning_voting_sparsity[pruning_voting_accuracy] = prune_size
        center_sparsity[center_accuracy] = center_prune_size

        logger.info(f'Pruning Voting Accuracy: {pruning_voting_accuracy}, Sparsity: {prune_size}%')
        logger.info(f'Federated Learning Accuracy: {federated_accuracy}, Sparsity: {federated_prune_size}%')
        logger.info(f'Center Model Accuracy: {center_accuracy}, Sparsity: {center_prune_size}%')

        plot_results(pruning_voting_sparsity, federated_sparsity, center_sparsity, args.plot_title, output_dir)

    # Conclusion to the simulation
    logger.info('Pruning Voting Algorithm results are:')
    logger.info(json.dumps(pruning_voting_sparsity, indent=4))

    logger.info('Federated Learning Algorithm results are:')
    logger.info(json.dumps(federated_sparsity, indent=4))

    logger.info('Center Model Algorithm results are:')
    logger.info(json.dumps(center_sparsity, indent=4))


if __name__ == "__main__":
    main()
