from __future__ import division, print_function


import time, argparse, os
import numpy as np

import torch
import torch.nn.functional  as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

#TRAINING 
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay )

def set_gpu():
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "6"

if args.cuda: 
    set_gpu()
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train],labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # evaluate validation seperately and remove dropout for it 
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val],labels[idx_val])
    
    print('Epoch:{0:4d}'.format(epoch+1),
          'loss_train:{:.4f}'.format(loss_train.item()),
          'acc_train:{:.4f}'.format(acc_train.item()),
          'loss_val:{:.4f}'.format(loss_val),
          'acc_val:{:.4f}'.format(acc_val),
          'time:{}'.format(time.time()-t))

def test():
  model.eval()
  output = model(features,adj)
  loss_test = F.nll_loss(output[idx_test],labels[idx_test])
  acc_test = accuracy(output[idx_test],labels[idx_test])
  print("Test results:"
        "loss = {:.4f}".format(loss_test.item()),
        "accuracy = {:.4f}".format(acc_test.item()))
  
t_total = time.time()

#########
print(args)
########

for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished")
print("Total time elapsed: {:.4f}s".format(time.time()-t_total))

test()
