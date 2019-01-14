from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import scipy.sparse as sp
import torch.optim as optim
from torch.autograd import Variable

from utils_cora import accuracy, load_data_cora
from model import AGNN
import networkx as nx

import torch
import torch.nn.functional as F

import pickle as pkl

dataset_all = ['CiteSeer', 'Cora', 'Pubmed']


import sys

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)




def train():

    t_total = time.time()
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


def test():

    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])


    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=True,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of attention layers.')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args.cuda, torch.cuda.is_available())

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    name_dataset = "CiteSeer"

    dataset = name_dataset.lower()

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(
                                "/media/benamira/19793564030D4273/MCsBackup/3A/OMA/NGSA/project/pytorch-AGNN/data/" + name_dataset + "/raw/ind.{}.{}".format(
                    dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(
        "/media/benamira/19793564030D4273/MCsBackup/3A/OMA/NGSA/project/pytorch-AGNN/data/" + name_dataset + "/raw/ind.{}.test.index".format(
            dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]



    idx_test = test_idx_range.tolist()
    idx_test = range(idx_test[0], idx_test[-20])
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)



    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]



    idx_test = torch.LongTensor(np.array(idx_test))

    idx_val = torch.LongTensor(np.array(idx_val))

    idx_train = torch.LongTensor(np.array(idx_train))



    indices = np.where(~labels.any(axis=1))[0]

    from random import randint

    p = randint(0, 3)

    for i in indices:
        labels[i][p] = 1



    labels = torch.LongTensor(np.where(labels)[1])

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))










    # Model and optimizer
    model = AGNN(nfeat=features.shape[1],
                 nhid=args.hidden,
                 nclass=labels.max() + 1,
                 nlayers=args.layers,
                 dropout_rate=args.dropout_rate)


    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    train()
    test()
