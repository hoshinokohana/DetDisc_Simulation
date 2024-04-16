import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

batch_size = 32
digits = (0, 1)  # Pair of digits for blending
contrast = (0.1, 0.9)  # Contrast levels for each digit
proportion = (0.6, 0.4)  # Proportions for blending
n_images = 10000

from architecture import *
from util import log
from shading import create_blended_dataset

# Use argparse or set manually
batch_size = 32
digits = (2, 7)

contrast = (0.1, 0.9)

proportion = (0.6, 0.4)
n_images = 10000
def generate_data_loader(digit1, digit2, contrast1, contrast2, n_images, batch_size):
    blended_images = create_blended_dataset(digit1, digit2, contrast1, contrast2, n_images)
    labels = torch.tensor([1 if i < n_images // 2 else 0 for i in range(n_images)], dtype=torch.float32)
    dataset = TensorDataset(blended_images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import argparse
import os
import sys
import time

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True


batch_size = 32
digits = (2, 7)  # Pair of digits for blending
contrast = (0.1, 0.9)  # Contrast levels for each digit
proportion = (0.6, 0.4)  # Proportions for blending
n_images = 10000

from architecture import *
from util import log

#create the datasets
from shading import create_blended_dataset


def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def train(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch):
	# Create file for saving training progress
	train_prog_dir = './train_prog/'
	check_path(train_prog_dir)
	model_dir = train_prog_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	train_prog_fname = model_dir + 'epoch_' + str(epoch) + '.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch class_loss class_acc conf_loss conf\n')
	# Set to training mode
	encoder.train()
	class_out.train()
	conf_out.train()
	# Iterate over batches
	for batch_idx, (data, target) in enumerate(train_loader):
		# Batch start time
		start_time = time.time()
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
		# Scale signal
		signal = ((torch.rand(x.shape[0]) * (args.		x = x * signal.view(-1, 1, 1, 1)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Threshold image
		x = nn.Hardtanh()(x)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze()
		conf = conf_out(z).squeeze()
		# Classification loss
		class_loss_fn = torch.nn.BCELoss()
		class_loss = class_loss_fn(y_pred, y_targ.float())
		# Classification accuracy
		correct_preds = torch.eq(y_pred.round(), y_targ.float()).float()
		class_acc = correct_preds.mean().item() * 100.0
		# Confidence loss
		conf_loss_fn = torch.nn.BCELoss()
		conf_loss = conf_loss_fn(conf, correct_preds)
		# Combine losses and update model
		combined_loss = class_loss + conf_loss
		combined_loss.backward()
		optimizer.step()
		# Overall confidence 
		avg_conf = conf.mean().item() * 100.0
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report prgoress
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Class. Loss = ' + '{:.4f}'.format(class_loss.item()) + '] ' + \
					 '[Class. Acc. = ' + '{:.2f}'.format(class_acc) + '] ' + \
					 '[Conf. Loss = ' + '{:.4f}'.format(conf_loss.item()) + '] ' + \
					 '[Conf. = ' + '{:.2f}'.format(avg_conf) + '] ' + \
					 '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
			# Save progress to file
			train_prog_f.write(str(batch_idx) + ' ' +\
				               '{:.4f}'.format(class_loss.item()) + ' ' + \
				               '{:.2f}'.format(class_acc) + ' ' + \
				               '{:.4f}'.format(conf_loss.item()) + ' ' + \
				               '{:.2f}'.format(avg_conf) + '\n')
	train_prog_f.close()

def s1s2_test(args, encoder, class_out, conf_out, device, s1_loader, s2_loader, targ_	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	# Iterate over batches
	all_test_correct_preds = []
	all_test_conf = []
	for batch_idx, ((data_s1, target_s1), (data_s2, target_s2)) in enumerate(zip(s1_loader, s2_loader)):
		# Load data
		x_s1 = data_s1.to(device)
		x_s2 = data_s2.to(device)
		# Sample targets and set signal level for s1/s2
		y_targ = torch.rand(args.test_batch_size).round().to(device)
		targ_signal = torch.ones(args.test_batch_size).to(device) * targ_signal
		nontarg_signal = torch.ones(args.test_batch_size).to(device) * nontarg_signal
		s1_signal = (targ_signal * torch.logical_not(y_targ).float()) + (nontarg_signal * y_targ)
		s2_signal = (targ_signal * y_targ) + (nontarg_signal * torch.logical_not(y_targ).float())
		# Apply contrast scaling and sumperimpose images
		x_s1 = x_s1 * s1_signal.view(-1,1,1,1)
		x_s2 = x_s2 * s2_signal.view(-1,1,1,1)
		x, _ = torch.stack([x_s1, x_s2],0).max(0)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		x = x + (torch.randn(x.shape) * noise).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze()
		conf = conf_out(z).squeeze()
		# Collect responses
		# Correct predictions
		correct_preds = torch.eq(y_pred.round(), y_targ.float()).float()
		all_test_correct_preds.append(correct_preds.detach().cpu().numpy())
		# Confidence
		all_test_conf.append(conf.detach().cpu().numpy())
	# Average test accuracy and confidence
	all_test_correct_preds = np.concatenate(all_test_correct_preds)
	all_test_conf = np.concatenate(all_test_conf)
	avg_test_acc = np.mean(all_test_correct_preds) * 100.0
	avg_test_conf = np.mean(all_test_conf) * 100.0
	avg_test_conf_correct = np.mean(all_test_conf[all_test_correct_preds==1]) * 100.0
	avg_test_conf_incorrect = np.mean(all_test_conf[all_test_correct_preds==0]) * 100.0
	return avg_test_acc, avg_test_conf, avg_test_conf_correct, avg_test_conf_incorrect

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-batch-size', type=int, default=32)
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--	parser.add_argument('--	parser.add_argument('--	parser.add_argument('--noise_level', type=float, default=0.9, help='Noise level for training labels')
	parser.add_argument('--img_size', type=int, default=32)
	parser.add_argument('--latent_dim', type=int, default=100)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--log_interval', type=int, default=10)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	parser.add_argument('--	parser.add_argument('--	args = parser.parse_args()
		
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Training set loader
	train_loader = generate_data_loader(digit1, digit2, contrast1, contrast2, n_images, batch_size)
	test_loader = generate_data_loader(digit1, digit2, contrast1, contrast2, n_images // 10, batch_size)


	# Build model
	log.info('Building model...')
	encoder = Encoder(args).to(device)
	class_out = Class_out(args).to(device)
	conf_out = Conf_out(args).to(device)
	all_modules = nn.ModuleList([encoder, class_out, conf_out])

	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(all_modules.parameters(), lr=args.lr)

	# Train
	log.info('Training begins...')
	for epoch in range(1, args.epochs + 1):
		# Training loop
		train(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch)

	# Test
	log.info('Test...')
	# Evaluate without noise
	test_acc_noiseless, _, __, ___ = test(args, encoder, class_out, conf_out, device, test_loader, 	# Signal and noise values for test
	signal_test_vals = np.linspace(args.	noise_test_vals = np.linspace(args.	all_test_acc = []
	all_test_conf = []
	all_test_conf_correct = []
	all_test_conf_incorrect = []
	for n in range(noise_test_vals.shape[0]):
		all_signal_test_acc = []
		all_signal_test_conf = []
		all_signal_test_conf_correct = []
		all_signal_test_conf_incorrect = []
		for s in range(signal_test_vals.shape[0]):
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = test(args, encoder, class_out, conf_out, device, test_loader, 			all_signal_test_acc.append(test_acc)
			all_signal_test_conf.append(test_conf)
			all_signal_test_conf_correct.append(test_conf_correct)
			all_signal_test_conf_incorrect.append(test_conf_incorrect)
		all_test_acc.append(all_signal_test_acc)
		all_test_conf.append(all_signal_test_conf)
		all_test_conf_correct.append(all_signal_test_conf_correct)
		all_test_conf_incorrect.append(all_signal_test_conf_incorrect)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'test_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=noise_test_vals,
			 test_acc_noiseless=test_acc_noiseless,
			 all_test_acc=np.array(all_test_acc),
			 all_test_conf=np.array(all_test_conf),
			 all_test_conf_correct=np.array(all_test_conf_correct),
			 all_test_conf_incorrect=np.array(all_test_conf_incorrect))

if __name__ == '__main__':
	main()