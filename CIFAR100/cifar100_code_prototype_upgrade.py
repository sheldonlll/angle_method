import argparse
from torch import nn as nn
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy.random as npr
import numpy as np
import sys
from itertools import chain
import copy
import math
from resnet import ResNet18
from torch.utils.data import Dataset
import random
import pickle
device = torch.device("cuda")

class CustomDataset(Dataset):
    def __init__(self) -> None:
        super(CustomDataset, self).__init__()
        self.targets = None
        self.data = None

    def add_items(self, inputs, targets):
        if isinstance(self.targets, np.ndarray) == False:
            self.targets = np.array(targets)
            self.data = np.array(inputs)
        else:
            self.targets = np.append(self.targets, targets)
            self.data = np.append(self.data, inputs, axis = 0)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)


def build_new_train_dataset(args, train_dataset, trainset_permutation_inds):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            filter_button, class_nums_counter, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    batch_size = args.batch_size
    new_train_dataset = CustomDataset()
    print("building new train_dataset....")
    for batch_index, batch_start_idx in  enumerate(range(0, len(train_dataset.targets), batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_idx: batch_start_idx + batch_size]
        elem_to_remove = np.setdiff1d(batch_inds.tolist(), list(visited_indexes))
        batch_inds = np.setdiff1d(batch_inds.tolist(), elem_to_remove.tolist()).astype(int)
        inputs = train_dataset.data[batch_inds]
        targets = np.array(train_dataset.targets)[batch_inds]
        new_train_dataset.add_items(inputs, targets)
    
    train_dataset = new_train_dataset
    print(f"len new_train_dataset.targets-data: {len(train_dataset.targets)}-{len(train_dataset.data)} / {len(visited_indexes)}")
    del new_train_dataset
    
    if args.run_mode == "filter" and args.filter_button == True:
        # 保存从全部数据中筛选出来的下标至filtered_index_file_path
        torch.save(list(visited_indexes), args.filtered_index_file_path)
    return train_dataset


def update_epoch_accuracy_matrix_dataset1(test_acc, current_epoch_accuracy_detail_lst, train_dataset, trainset_permutation_inds, args):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            filter_button, class_nums_counter, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, th_win, th_train_acc, th_stable_acc, th_stable_loss, ordered_values, ordered_indx
    
    if threshold_epoch != None and test_acc >= threshold_test_accuracy_to_build_new_data and filter_button == True:
        filter_button = False
        epoch_accuracy_matrix = np.array(epoch_accuracy_matrix).T
        
        ratio = (min_max_data_nums_per_class[1] * 100 / epoch_accuracy_matrix.shape[0])
        visited_indexes = set()
        pop_num = 10
        cnt_almost_same_times = 0
        max_almost_same_times = 10
        min_delta_same_num = 5
        last_len_vis = 0
        while len(visited_indexes) < int(min_max_data_nums_per_class[1] * 100):

            start_idx = random.randint(0, epoch_accuracy_matrix.shape[0] - 1)
            end_idx = min(start_idx + 127,  epoch_accuracy_matrix.shape[0] - 1)
            compare_start_idx = random.randint(0, epoch_accuracy_matrix.shape[0] - 1)
            compare_end_idx = min(compare_start_idx + 127,  epoch_accuracy_matrix.shape[0] - 1)
            
            compare_matrix = torch.tensor(epoch_accuracy_matrix[compare_start_idx: compare_end_idx + 1].astype(float)).to(device).unsqueeze(1)
            select_matrix = torch.tensor(epoch_accuracy_matrix[start_idx: end_idx + 1].astype(float)).to(device).unsqueeze(0)
            current_angles = torch.acos(torch.cosine_similarity(compare_matrix, select_matrix, dim=-1))
            sorted_angles = np.array([sorted([(current_angles[me_index][other_index].detach().item(), \
                                            trainset_permutation_inds[start_idx + other_index], trainset_permutation_inds[compare_start_idx + me_index])\
                                            for other_index in range(current_angles.shape[1])], key = lambda x: x[0]) for me_index in range(current_angles.shape[0])])
            max_num = int(sorted_angles.shape[0] * ratio)
            for i in range(sorted_angles.shape[0]):
                for k in range(sorted_angles[i].shape[0] - 1, 0, -1):
                    right_angle, right_other, right_me = sorted_angles[i][k]
                    right_other, right_me = int(right_other), int(right_me)
                    if k < max_num: break
                    if right_other in unforgettable_idx and right_me not in unforgettable_idx:
                        if len(visited_indexes) < int(min_max_data_nums_per_class[1] * 100): visited_indexes.add(right_me) 
                    elif right_other not in unforgettable_idx and right_me in unforgettable_idx:
                        if len(visited_indexes) < int(min_max_data_nums_per_class[1] * 100): visited_indexes.add(right_other) 
                    else: break
            print(f"len visited_indexes: {len(visited_indexes)} / {int(min_max_data_nums_per_class[1] * 100)}")
            if int(abs(last_len_vis - len(visited_indexes))) < min_delta_same_num:
                cnt_almost_same_times += 1
            if cnt_almost_same_times > max_almost_same_times:
                cnt_almost_same_times = 0
                for i in range(min(pop_num, len(unforgettable_idx))):
                    unforgettable_idx.pop()
                print(f"pop unforgettable_idx {pop_num}, len unforgettable_idx: {len(unforgettable_idx)}")
            last_len_vis = len(visited_indexes)
        epoch_accuracy_matrix = epoch_accuracy_matrix[list(visited_indexes)].T.tolist()
        current_epoch_accuracy_detail_lst = np.array(list(chain(*current_epoch_accuracy_detail_lst)))[list(visited_indexes)]

        epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)
        print(f"append finished, epoch_accuracy_matrix shape: {np.array(epoch_accuracy_matrix).shape}\n")
        train_dataset = build_new_train_dataset(args, train_dataset, trainset_permutation_inds)

    else:
        current_epoch_accuracy_detail_lst = list(chain(*current_epoch_accuracy_detail_lst))
        epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)
        print("\n")

def update_epoch_accuracy_matrix_dataset(test_acc, current_epoch_accuracy_detail_lst, train_dataset, trainset_permutation_inds, args):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            filter_button, class_nums_counter, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, th_win, th_train_acc, th_stable_acc, th_stable_loss, ordered_values, ordered_indx
    
    if threshold_epoch != None and test_acc >= threshold_test_accuracy_to_build_new_data and filter_button == True:
        filter_button = False
        epoch_accuracy_matrix = np.array(epoch_accuracy_matrix).T
        
        ratio = (min_max_data_nums_per_class[1] * 100 / epoch_accuracy_matrix.shape[0])
        visited_indexes = set()
        while len(visited_indexes) < int(min_max_data_nums_per_class[1] * 100):

            start_idx = random.randint(0, epoch_accuracy_matrix.shape[0] - 1)
            end_idx = min(start_idx + 127,  epoch_accuracy_matrix.shape[0] - 1)
            compare_start_idx = random.randint(0, epoch_accuracy_matrix.shape[0] - 1)
            compare_end_idx = min(compare_start_idx + 127,  epoch_accuracy_matrix.shape[0] - 1)
            
            compare_matrix = torch.tensor(epoch_accuracy_matrix[compare_start_idx: compare_end_idx + 1].astype(float)).to(device).unsqueeze(1)
            select_matrix = torch.tensor(epoch_accuracy_matrix[start_idx: end_idx + 1].astype(float)).to(device).unsqueeze(0)
            current_angles = torch.acos(torch.cosine_similarity(compare_matrix, select_matrix, dim=-1))
            sorted_angles = np.array([sorted([(current_angles[me_index][other_index].detach().item(), \
                                            trainset_permutation_inds[start_idx + other_index], trainset_permutation_inds[compare_start_idx + me_index])\
                                            for other_index in range(current_angles.shape[1])], key = lambda x: x[0]) for me_index in range(current_angles.shape[0])])
            max_num = int(sorted_angles.shape[0] * ratio)
            for i in range(sorted_angles.shape[0]):
                for k in range(sorted_angles[i].shape[0] - 1, 0, -1):
                    right_angle, right_other, right_me = sorted_angles[i][k]
                    right_other, right_me = int(right_other), int(right_me)
                    if k < max_num: break
                    if right_other in unforgettable_idx and right_me not in unforgettable_idx:
                        if len(visited_indexes) < int(min_max_data_nums_per_class[1] * 100): visited_indexes.add(right_me) 
                    elif right_other not in unforgettable_idx and right_me in unforgettable_idx:
                        if len(visited_indexes) < int(min_max_data_nums_per_class[1] * 100): visited_indexes.add(right_other) 
                    else: break
            print(f"len visited_indexes: {len(visited_indexes)} / {int(min_max_data_nums_per_class[1] * 100)}")
        
        epoch_accuracy_matrix = epoch_accuracy_matrix[list(visited_indexes)].T.tolist()
        current_epoch_accuracy_detail_lst = np.array(list(chain(*current_epoch_accuracy_detail_lst)))[list(visited_indexes)]

        epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)
        print(f"append finished, epoch_accuracy_matrix shape: {np.array(epoch_accuracy_matrix).shape}\n")
        train_dataset = build_new_train_dataset(args, train_dataset, trainset_permutation_inds)

    else:
        current_epoch_accuracy_detail_lst = list(chain(*current_epoch_accuracy_detail_lst))
        epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)
        print("\n")


def update_threshold_epoch(epoch):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            filter_button, class_nums_counter, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    if len(train_acc_list) > th_win and threshold_epoch == None:
        temp_acc_train_list, temp_loss_train_list = sorted(train_acc_list[- th_win - 1: -1]), sorted(train_loss_list[- th_win - 1: -1])
        min_acc_train, max_acc_train, mediean_acc_train = temp_acc_train_list[0], temp_acc_train_list[-1], sum(temp_acc_train_list) / th_win
        min_loss_train, max_loss_train = temp_loss_train_list[0], temp_loss_train_list[-1]
        delta_acc_train = max_acc_train - min_acc_train
        delta_loss_train = max_loss_train - min_loss_train
        print(f"delta_acc_train: {delta_acc_train}\ndelta_loss_train: {delta_loss_train}\nmin_loss_train: {min_loss_train}\nmax_loss_train: {max_loss_train}\nmin_acc_train: {min_acc_train}\nmax_acc_train: {max_acc_train}\nmediean_acc_train: {mediean_acc_train}\n")
        if mediean_acc_train > th_train_acc:
            threshold_epoch = epoch
            print("acc")
        elif delta_acc_train < th_stable_acc and delta_loss_train < th_stable_loss:
            threshold_epoch = epoch
            print("delta")


def train_one_epoch(args, model, device, train_dataset, optimizer, criterion, epoch):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            filter_button, class_nums_counter, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    train_loss = 0.
    correct = 0.
    total = 0.

    model.train()

    trainset_permutation_inds = npr.permutation(np.arange(len(train_dataset.targets)))
    
    batch_size = args.batch_size
    
    # init
    current_epoch_accuracy_detail_lst = []
    
    batch_cnt = 0
    for batch_index, batch_start_idx in  enumerate(range(0, len(train_dataset.targets), batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_idx: batch_start_idx + batch_size]
        batch_cnt += 1

        transformed_train_dataset = []
        for ind in batch_inds:
            transformed_train_dataset.append(train_dataset.__getitem__(ind)[0])
        
        inputs = torch.stack(transformed_train_dataset).to(device)
        targets = torch.LongTensor(np.array(train_dataset.targets)[batch_inds].tolist()).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        

        # update model parameters, accuracy, loss
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        result = predicted.eq(targets.data).cpu()
        correct += result.sum()
        current_epoch_accuracy_detail_lst.append(result.tolist())
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write(
            '| Train | Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_index + 1,
             (len(train_dataset) // batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()
    
    if isinstance(correct, float) == False:
        correct = correct.item()
    return correct / total, train_loss / batch_cnt, current_epoch_accuracy_detail_lst, trainset_permutation_inds


def test_one_epoch(args, epoch, model, device, test_dataset, criterion):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            filter_button, class_nums_counter, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, th_win, th_train_acc, th_stable_acc, th_stable_loss
    test_loss = 0.
    correct = 0.
    total = 0.
    test_batch_size = 32

    model.eval()
    batch_cnt = 0
    for batch_index, batch_start_ind in enumerate(range(0, len(test_dataset.targets), test_batch_size)):
        batch_cnt += 1

        transformed_testset = []
        for ind in range(batch_start_ind, min(len(test_dataset.targets), batch_start_ind + test_batch_size)):
            transformed_testset.append(test_dataset.__getitem__(ind)[0])
        
        inputs = torch.stack(transformed_testset).to(device)
        targets = torch.LongTensor(np.array(test_dataset.targets)[batch_start_ind:batch_start_ind + test_batch_size].tolist()).to(device)

        # Forward propagation, compute loss, get predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Test | Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_index + 1,
             (len(test_dataset) // test_batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()
    
    if isinstance(correct, float) == False:
        correct = correct.item()
    return correct / total, test_loss / batch_cnt
    
train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
threshold_epoch = None
visited_indexes = set()
min_max_data_nums_per_class = []
threshold_occupation = 0
epoch_accuracy_matrix = []
sorted_result_angles = []
filter_button = False
has_calculate_forget_events = False
class_nums_counter = {class_name:0 for class_name in range(10)}
start_new_dataloader_min_nums = 0
threshold_test_accuracy_to_build_new_data = 0
n_samples, *n_features = 0, 0
th_win = 0
th_train_acc = 0
th_stable_acc = 0
th_stable_loss = 0
unforgettable_idx = set()
ordered_values = None
ordered_indx = None
def main(args):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, th_win, th_train_acc, th_stable_acc, th_stable_loss, ordered_values, ordered_indx
    # model, optim, criterion, scheduler
    model = ResNet18(num_classes = 100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduce=False)
    scheduler = MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    # load dataset
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4)) # data augmentation
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR100(
        root='/tmp/data/',
        train=True,
        transform=train_transform,
        download=True)
    test_dataset = datasets.CIFAR100(
        root='/tmp/data/',
        train=False,
        transform=test_transform,
        download=True)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    npr.seed(args.seed)
    random.seed(args.seed)

    train_indx = None
    if args.run_mode == "filter":
        train_indx = np.array(range(len(train_dataset.targets)))
        train_dataset.data = train_dataset.data[train_indx, :, :, :]
        train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()
        with open(args.sorting_file, 'rb') as fin:
            items = list(pickle.load(fin).items())
            ordered_indx = items[0][-1]
            ordered_values = items[1][-1]
        non_zero_forget_count_index = (np.array(ordered_values) > 0).argmax(axis = 0) # 从有序forget count列表中找到第一个非零元素的下标
        unforgettable_idx = set(np.array(ordered_indx)[:non_zero_forget_count_index]) # 从forget中load unforget examples
        print(f"forget-len(unforgettable_idx): {len(unforgettable_idx)}")
    elif args.run_mode == "pure":
        train_indx = torch.load(args.filtered_index_file_path)
        train_dataset.data = train_dataset.data[train_indx, :, :, :]
        train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()
        print(f"len train_dataset.data: {len(train_dataset.data)}")

    # init
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    threshold_epoch = None
    visited_indexes = set()
    min_max_data_nums_per_class = [args.min_data_nums_per_class, args.max_data_nums_per_class]
    threshold_occupation = args.threshold_occupation
    epoch_accuracy_matrix = []
    sorted_result_angles = []
    filter_button = args.filter_button if args.run_mode == "filter" else False
    class_nums_counter = {class_name:0 for class_name in range(10)}
    start_new_dataloader_min_nums = args.start_new_dataloader_min_nums
    threshold_test_accuracy_to_build_new_data = args.threshold_test_accuracy_to_build_new_data
    n_samples, *n_features = train_dataset.data.shape
    th_win = args.th_win
    th_train_acc = args.th_train_acc
    th_stable_acc = args.th_stable_acc
    th_stable_loss = args.th_stable_loss
    print(f"filter_button: {filter_button}, args.filter_button: {args.filter_button}")

    # train test loop
    for epoch in range(args.epochs):
        train_acc, train_loss, current_epoch_accuracy_detail_lst, trainset_permutation_inds = train_one_epoch(args, model, device, train_dataset, optimizer, criterion, epoch)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        test_acc, test_loss = test_one_epoch(args, epoch, model, device, test_dataset, criterion)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)


        if filter_button == True: 
            update_epoch_accuracy_matrix_dataset(test_acc, current_epoch_accuracy_detail_lst, train_dataset, trainset_permutation_inds, args)

        if threshold_epoch == None:
            update_threshold_epoch(epoch)

        print(f"epoch: {epoch}, acc_train: {train_acc_list[-1]}, loss_train: {train_loss_list[-1]}, acc_test: {test_acc_list[-1]}, loss_test: {test_loss_list[-1]}")

        scheduler.step()

    print(f"train_acc_list: {train_acc_list}")
    print(f"train_loss_list: {train_loss_list}")
    print(f"test_acc_list: {test_acc_list}")
    print(f"test_loss_list: {test_loss_list}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument("--epochs", type=int, default = 200, help = "epochs")
    parser.add_argument("--batch_size", type=int, default = 128, help = "batch_size for training")
    parser.add_argument("--lr", type=float, default = 0.1, help = "lr")
    parser.add_argument("--seed", type=int, default = 1, help = "seed")
    parser.add_argument("--th_win", type=int, default = 3, help = "th_win")
    parser.add_argument("--th_stable_acc", type=float, default = 1e-3, help = "th_stable_acc")
    parser.add_argument("--th_stable_loss", type=float, default = 1e-3, help = "th_stable_loss")
    parser.add_argument("--th_train_acc", type=float, default = 0.85, help = "th_train_acc")
    parser.add_argument("--filter_button", action = "store_true", help="filter_button")
    parser.add_argument("--threshold_occupation", type=int, default=2, help="threshold_occupation")
    parser.add_argument("--start_new_dataloader_min_nums", type=int, default=100, help="start_new_dataloader_min_nums")
    parser.add_argument("--threshold_test_accuracy_to_build_new_data", type=float, default=0.6, help="threshold_test_accuracy_to_build_new_data")
    parser.add_argument("--min_data_nums_per_class", type=int, default=10, help="min_data_nums_per_class")
    parser.add_argument("--max_data_nums_per_class", type=int, default=2000, help="max_data_nums_per_class")
    parser.add_argument("--run_mode", type=str, default="filter", help="run_mode") 
    '''
    run_mode:
        "filter": 1.全部数据在转折点之前跑,forget计算原型,angle根据unforgettable example对应挑选,用挑选后的数据跑转折点之后, 
                2.并保存从全部数据中筛选出来的下标至filtered_index_file_path，当前shuffle序列保存至filtered_index_trainset_permutation_inds_file_path
        "pure": 用filter保存下来的下标,在训练之前将原始数据变成指定的数据,用这些数据跑完所有epochs
        全部数据跑完所有epoch: 将filter_button设为False即可
    '''
    parser.add_argument("--filtered_index_file_path", type=str, default="C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\filtered_index_file_path.pt", help="filtered_index_file_path")
    parser.add_argument("--sorting_file", type=str, default="C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\cifar10_results\\cifar10_sorted.pkl", help="sorting_file")

    args = parser.parse_args()
    main(args)

# filter:
# python cifar100_code_prototype_upgrade.py --epochs 200 --batch_size 128 --lr 0.1 --seed 1 --th_win 3 --th_stable_acc 0.001 --th_stable_loss 0.001 --th_train_acc 0.85 --filter_button --threshold_occupation 2 --start_new_dataloader_min_nums 100 --threshold_test_accuracy_to_build_new_data 0.6  --min_data_nums_per_class 10 --max_data_nums_per_class 2000 --run_mode filter --filtered_index_file_path C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\filtered_index_file_path100_20.pt --sorting_file C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\cifar100_results\\cifar100_sorted.pkl