from __future__ import print_function
import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms


# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Train model for one epoch
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def train(args, model, device, trainset, optimizer, epoch, example_stats):
    train_loss = 0
    correct = 0
    total = 0
    batch_size = args.batch_size

    model.train()

    # Get permutation to shuffle trainset
    trainset_permutation_inds = npr.permutation(
        np.arange(len(trainset.train_labels)))

    for batch_idx, batch_start_ind in enumerate(
            range(0, len(trainset.train_labels), batch_size)):

        # Get trainset indices for batch
        batch_inds = trainset_permutation_inds[batch_start_ind:
                                               batch_start_ind + batch_size]

        # Get batch inputs and targets, transform them appropriately
        transformed_trainset = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(
            np.array(trainset.train_labels)[batch_inds].tolist())

        # Map to available device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward propagation, compute loss, get predictions
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)

        # Update statistics and loss
        acc = predicted == targets
        for j, index in enumerate(batch_inds):

            # Get index in original dataset (not sorted by forgetting)
            index_in_original_dataset = train_indx[index]

            # Compute missclassification margin
            output_correct_class = outputs.data[
                j, targets[j].item()]  # output for correct class
            sorted_output, _ = torch.sort(outputs.data[j, :])
            if acc[j]:
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_output[-2]
            else:
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(
            ) - output_highest_incorrect_class.item()

            # Add the statistics of the current training example to dictionary
            index_stats = example_stats.get(index_in_original_dataset,
                                            [[], [], []])
            index_stats[0].append(loss[j].item())
            index_stats[1].append(acc[j].sum().item())
            index_stats[2].append(margin)
            example_stats[index_in_original_dataset] = index_stats

        # Update loss, backward propagate, update optimizer
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_idx + 1,
             (len(trainset) // batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()

        # Add training accuracy to dict
        index_stats = example_stats.get('train', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats['train'] = index_stats


# Evaluate model predictions on heldout test data
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def test(args, model, device, testset, example_stats):
    test_loss = 0
    correct = 0
    total = 0
    test_batch_size = 32

    model.eval()

    for batch_idx, batch_start_ind in enumerate(
            range(0, len(testset.test_labels), test_batch_size)):

        # Get batch inputs and targets
        transformed_testset = []
        for ind in range(
                batch_start_ind,
                min(
                    len(testset.test_labels),
                    batch_start_ind + test_batch_size)):
            transformed_testset.append(testset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_testset)
        targets = torch.LongTensor(
            np.array(testset.test_labels)[batch_start_ind:batch_start_ind +
                                          test_batch_size].tolist())

        # Map to available device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward propagation, compute loss, get predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Add test accuracy to dict
    acc = 100. * correct.item() / total
    index_stats = example_stats.get('test', [[], []])
    index_stats[1].append(100. * correct.item() / float(total))
    example_stats['test'] = index_stats
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
          (epoch, loss.item(), acc))


parser = argparse.ArgumentParser(description='training MNIST')
parser.add_argument(
    '--dataset',
    default='fashionmnist',
    help='dataset to use, can be mnist or permuted_mnist')
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    metavar='N',
    help='number of epochs to train (default: 200)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--no_cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--sorting_file',
    default="none",
    help=
    'name of a file containing order of examples sorted by a certain metric (default: "none", i.e. not sorted)'
)
parser.add_argument(
    '--remove_n',
    type=int,
    default=0,
    help='number of sorted examples to remove from training')
parser.add_argument(
    '--keep_lowest_n',
    type=int,
    default=0,
    help=
    'number of sorted examples to keep that have the lowest metric score, equivalent to start index of removal; if a negative number given, remove random draw of examples'
)
parser.add_argument(
    '--no_dropout', action='store_true', default=False, help='remove dropout')
parser.add_argument(
    '--input_dir',
    default='fashionmnist_results/',
    help='directory where to read sorting file from')
parser.add_argument(
    '--output_dir', required=True, help='directory where to save results')

# Enter all arguments that you want to be in the filename of the saved output
ordered_args = [
    'dataset', 'no_dropout', 'seed', 'sorting_file', 'remove_n',
    'keep_lowest_n'
]
def to_win(path):
    path = path.split("\\")
    path = '/'.join(path)
    return path
# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()
args.output_dir = to_win(args.output_dir)
args_dict = vars(args)
print(args_dict)
save_fname = '__'.join(
    '{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

# Set appropriate devices
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Set random seed for initialization
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
npr.seed(args.seed)

# Setup transforms
all_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]
transform = transforms.Compose(all_transforms)

os.makedirs(args.output_dir, exist_ok=True)

# Load the appropriate train and test datasets
trainset = datasets.FashionMNIST(
    root='/tmp/data', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(
    root='/tmp/data', train=False, download=True, transform=transform)

# Get indices of examples that should be used for training
if args.sorting_file == 'none':
    train_indx = np.array(range(len(trainset.train_labels)))
else:
    try:
        with open(
                os.path.join(args.input_dir, args.sorting_file) + '.pkl',
                'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']
    except IOError:
        with open(os.path.join(args.input_dir, args.sorting_file),
                  'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']

    # Get the indices to remove from training
    elements_to_remove = np.array(
        ordered_indx)[args.keep_lowest_n:args.keep_lowest_n + args.remove_n]

    # Remove the corresponding elements
    train_indx = np.setdiff1d(
        range(len(trainset.train_labels)), elements_to_remove)

# Remove remove_n number of examples from the train set at random
if args.keep_lowest_n < 0:
    train_indx = npr.permutation(np.arange(len(
        trainset.train_labels)))[:len(trainset.train_labels) - args.remove_n]

# Reassign train data and labels
trainset.data = trainset.data[train_indx, :, :]
trainset.targets = np.array(trainset.targets)[train_indx].tolist()

print('Training on ' + str(len(trainset.train_labels)) + ' examples')

# Setup model and optimizer
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Setup loss
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce=False)

# Initialize dictionary to save statistics for every example presentation
example_stats = {}

elapsed_time = 0
for epoch in range(args.epochs):
    start_time = time.time()

    train(args, model, device, trainset, optimizer, epoch, example_stats)
    test(args, model, device, testset, example_stats)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    # Save the stats dictionary
    fname = args.output_dir + "/" + save_fname
    with open(fname + "__stats_dict.pkl", "wb") as f:
        pickle.dump(example_stats, f)

    # Log the best train and test accuracy so far
    with open(fname + "__best_acc.txt", "w") as f:
        f.write('train test \n')
        f.write(str(max(example_stats['train'][1])))
        f.write(' ')
        f.write(str(max(example_stats['test'][1])))