from utils.ArticlesHandler import ArticlesHandler
from utils import Config
import time
import numpy as np
from pygcn.utils import accuracy, load_from_features
from pyagnn.agnn.model import AGNN
import torch
import torch.nn.functional as F
import torch.optim as optim

config = Config(file='config')

assert (config.num_fake_articles + config.num_real_articles >
        config.num_nearest_neighbours), "Can't have more neighbours than nodes!"

print("Method of decomposition:", config.method_decomposition_embedding)

print("Loading dataset", config.dataset_name)
articles = ArticlesHandler(config)

print("Performing decomposition...")
C = articles.get_tensor()

labels = articles.articles.labels
all_labels = articles.articles.labels_untouched

adj, features, all_labels = load_from_features(C, all_labels, config)
_, _, labels = load_from_features(C, labels, config)

nbre_total_article = config['num_real_articles'] + config['num_fake_articles']
pourcentage_know = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
pourcentage_voisin = np.array([1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 15])
ratios = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
methods = [("decomposition", False),  ("GloVe", "mean"), ("",""),("","")]

idx_train = np.where(labels)[0]
idx_val = np.where(1 - abs(labels))[0][:90]
idx_test = np.where(1 - abs(labels))[0][90:]

print(len(idx_train))

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

cuda = False
hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
fastmode = False
epochs = 450#450
layers =2

# Model and optimizer
model = AGNN(nfeat=features.shape[1],
                     nhid=hidden,
                     nclass=2,
                     nlayers=layers,
                     dropout_rate=0.5)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    all_labels = all_labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

t_total = time.time()
for epoch in range(epochs):

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    loss_train = F.nll_loss(output[idx_train], all_labels[idx_train])
    acc_train = accuracy(output[idx_train], all_labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], all_labels[idx_val])
    acc_val = accuracy(output[idx_val], all_labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

model.eval()
output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], all_labels[idx_test])
acc_test = accuracy(output[idx_test], all_labels[idx_test])
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test.item()))