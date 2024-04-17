import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from copy import deepcopy
import argparse
import os
import sys
import time

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from architecture import *
from util import log
from shading import create_blended_digit_dataset, load_mnist_data  # Importing functions from shading.py


def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def train(args, encoder, class_out, conf_out, device, train_loader, optimizer, criterion, epoch):
    train_prog_dir = './train_prog/'
    check_path(train_prog_dir)
    model_dir = train_prog_dir + 'run' + str(epoch) + '/'
    check_path(model_dir)
    train_prog_fname = model_dir + 'epoch_' + str(epoch) + '.txt'
    
    with open(train_prog_fname, 'w') as train_prog_f:
        train_prog_f.write('batch class_loss class_acc conf_correct conf_incorrect\\n')
        
        # Debug: Check types before training
        print("Debug: Type of encoder:", type(encoder))
        print("Debug: Type of class_out:", type(class_out))
        print("Debug: Type of conf_out:", type(conf_out))

        # Set models to training mode
        encoder.train()
        class_out.train()
        conf_out.train()
        
        # Iterate over batches
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()
            data, target = data.to(device), target.to(device)
            
            # Zero out gradients for optimizer
            optimizer.zero_grad()
            
            # Forward pass through the encoder and classification/confidence heads
            z = encoder(data, device)
            y_pred = class_out(z)
            conf = conf_out(z)
            
            # Calculate losses
            class_loss = criterion(y_pred, target.float().view(-1, 1))  # Ensure target is correct shape
            correct_preds = (y_pred.round() == target.float().view(-1, 1)).float()
            class_acc = correct_preds.mean().item() * 100
            conf_loss = criterion(conf, correct_preds)  # Using same loss function for simplicity
            
            # Combine losses and update model parameters
            combined_loss = class_loss + conf_loss
            combined_loss.backward()
            optimizer.step()
            
            # Calculate confidence for correct and incorrect classifications
            conf_correct = conf[correct_preds.bool()].mean().item() * 100
            conf_incorrect = conf[~correct_preds.bool()].mean().item() * 100 if (~correct_preds.bool()).any() else 0
            
            end_time = time.time()
            batch_duration = end_time - start_time
            
            if batch_idx % 10 == 0:
                log_info = f'[Epoch: {epoch}] [Batch: {batch_idx}/{len(train_loader)}] ' + \
                           f'[Class Loss: {class_loss.item():.4f}] [Class Acc: {class_acc:.2f}%] ' + \
                           f'[Conf Correct: {conf_correct:.2f}%] [Conf Incorrect: {conf_incorrect:.2f}%] ' + \
                           f'[{batch_duration:.3f} sec/batch]'
                log.info(log_info)
                train_prog_f.write(f'{batch_idx} {class_loss.item():.4f} {class_acc:.2f} {conf_correct:.2f} {conf_incorrect:.2f}\n')



def s1s2_test(args, encoder, class_out, conf_out, device, s1_loader, s2_loader):
    # Set to evaluation mode
    encoder.eval()
    class_out.eval()
    conf_out.eval()

    # Iterate over batches
    all_test_correct_preds = []
    all_test_conf_correct = []
    all_test_conf_incorrect = []
    
    for batch_idx, ((data_s1, target_s1), (data_s2, target_s2)) in enumerate(zip(s1_loader, s2_loader)):
        # Load data
        x_s1 = data_s1.to(device)
        x_s2 = data_s2.to(device)
        
        # Get model predictions for both datasets
        z_s1 = encoder(x_s1, device)
        y_pred_s1 = class_out(z_s1).squeeze()
        conf_s1 = conf_out(z_s1).squeeze()

        z_s2 = encoder(x_s2, device)
        y_pred_s2 = class_out(z_s2).squeeze()
        conf_s2 = conf_out(z_s2).squeeze()

        # Calculate metrics for s1
        correct_preds_s1 = (y_pred_s1.round() == target_s1.float().to(device)).float()
        all_test_correct_preds.append(correct_preds_s1.detach().cpu().numpy())
        all_test_conf_correct.extend(conf_s1[correct_preds_s1.bool()].detach().cpu().numpy())
        all_test_conf_incorrect.extend(conf_s1[~correct_preds_s1.bool()].detach().cpu().numpy())

        # Calculate metrics for s2
        correct_preds_s2 = (y_pred_s2.round() == target_s2.float().to(device)).float()
        all_test_correct_preds.append(correct_preds_s2.detach().cpu().numpy())
        all_test_conf_correct.extend(conf_s2[correct_preds_s2.bool()].detach().cpu().numpy())
        all_test_conf_incorrect.extend(conf_s2[~correct_preds_s2.bool()].detach().cpu().numpy())

    # Aggregate all results
    all_test_correct_preds = np.concatenate(all_test_correct_preds)
    avg_test_acc = np.mean(all_test_correct_preds) * 100.0
    avg_test_conf_correct = np.mean(all_test_conf_correct) * 100.0 if all_test_conf_correct else 0
    avg_test_conf_incorrect = np.mean(all_test_conf_incorrect) * 100.0 if all_test_conf_incorrect else 0

    return avg_test_acc, avg_test_conf_correct, avg_test_conf_incorrect

def main():
    # Settings
    parser = argparse.ArgumentParser(description='Train and evaluate the model on a blended MNIST dataset')
    parser.add_argument('--digit1', type=int, default=1, help='First digit to blend')
    parser.add_argument('--digit2', type=int, default=0, help='Second digit to blend')
    parser.add_argument('--n_images', type=int, default=1000, help='Number of images to blend for each digit')
    parser.add_argument('--contrast1', type=float, default=0.2, help='Contrast adjustment for the first digit')
    parser.add_argument('--contrast2', type=float, default=0.8, help='Contrast adjustment for the second digit')
    parser.add_argument('--train-batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum for the optimizer')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensionality of the latent space')
    parser.add_argument('--no_cuda', action='store_true', help='Disables CUDA training')
    parser.add_argument('--run', type=int, default=1, help='Run identifier')
    args = parser.parse_args()


    # Set up cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    # Determine CUDA usage based on the no_cuda flag and CUDA availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Load MNIST data and create blended dataset
    mnist_data = load_mnist_data()
    blended_data = create_blended_digit_dataset(mnist_data, args.digit1, args.digit2, args.n_images, args.contrast1, args.contrast2)

    # Generate labels for the blended dataset
    # Assuming labels should be 0 for digit1 and 1 for digit2, or vice versa
    labels = torch.zeros(blended_data.size(0), dtype=torch.long)
    labels[args.n_images:] = 1  # Assuming first half is digit1, second half is digit2

    # Splitting blended dataset into train and test sets
    # This is a simple split; adjust the ratio as necessary
    split_index = int(0.8 * len(blended_data))
    train_data, test_data = blended_data[:split_index], blended_data[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup (assuming Net() and other models are correctly defined elsewhere)
# Build model
    log.info('Building model...')
    encoder = Encoder(args).to(device)
    class_out = Class_out(args).to(device)
    conf_out = Conf_out(args).to(device)
    all_modules = nn.ModuleList([encoder, class_out, conf_out])

	# Create optimizer
    log.info('Setting up optimizer...')
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': class_out.parameters()}, {'params': conf_out.parameters()}], lr=0.001)

    # Define the loss criterion
    criterion = torch.nn.CrossEntropyLoss()  # Or another appropriate loss function

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, encoder, class_out, conf_out, device, train_loader, optimizer, criterion, epoch)


    # Save test results
    test_dir = './test/'
    check_path(test_dir)
    model_dir = f"{test_dir}run{args.run}/"
    check_path(model_dir)
    test_acc, test_conf,test_conf_correct, test_conf_incorrect = s1s2_test(encoder, device, test_loader, criterion)

    np.savez(model_dir + 'test_results.npz', test_acc=test_acc, test_conf=test_conf,
             test_conf_correct=test_conf_correct, test_conf_incorrect=test_conf_incorrect)


if __name__ == '__main__':
	main()