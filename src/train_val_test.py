import os
import time
from argparse import ArgumentParser

import torch
from torch import nn
from tqdm import tqdm

from model.data import Data
from model.metric import Metric
from model.model import Model

parser = ArgumentParser()
parser.add_argument('--data_dir')
parser.add_argument('--save_dir')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

data = Data(args.data_dir, batch_size=128, num_workers=0)
train_loader, val_loader, test_loader = data.train_loader(), data.val_loader(), data.test_loader()
model = Model(len(data.vocab.token_to_idx), 128, 100).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_metric, eval_metric = Metric(), Metric()

best = 0.
for epoch in range(1, 6):
    print(f'epoch: {epoch}')

    model.train()
    train_metric.reset()
    loss, forward_time = 0., 0.
    for (features1, features2), labels in tqdm(train_loader, desc='train'):
        labels = labels.to(device)

        start_time = time.time()
        outputs = model(features1, features2)
        end_time = time.time()

        ls = criterion(outputs, labels)
        train_metric.update(torch.sigmoid(outputs), labels)

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        loss += ls.item() * labels.size(0)
        forward_time += end_time - start_time
    print(f'train loss: {(loss / len(train_loader.dataset)):.4f}\t'
          f'forward_time: {(1e6 * forward_time / len(train_loader.dataset)):.4f}e-6\t', end='')
    train_metric.compute()

    model.eval()
    eval_metric.reset()
    loss, forward_time = 0., 0.
    for (features1, features2), labels in tqdm(val_loader, desc='val'):
        labels = labels.to(device)

        start_time = time.time()
        outputs = model(features1, features2)
        end_time = time.time()

        ls = criterion(outputs, labels)
        eval_metric.update(torch.sigmoid(outputs), labels)

        loss += ls.item() * labels.size(0)
        forward_time += end_time - start_time
    print(f'val loss: {(loss / len(val_loader.dataset)):.4f}\t'
          f'forward_time: {(1e6 * forward_time / len(val_loader.dataset)):.4f}e-6\t', end='')
    cur = eval_metric.compute()
    if cur > best:
        torch.save(model, os.path.join(args.save_dir, 'best.pt'))
        best = cur

model = torch.load(os.path.join(args.save_dir, 'best.pt'), map_location=device)
model.eval()
eval_metric.reset()
loss, forward_time = 0., 0.
for (features1, features2), labels in tqdm(test_loader, desc='test'):
    labels = labels.to(device)

    start_time = time.time()
    outputs = model(features1, features2)
    end_time = time.time()

    ls = criterion(outputs, labels)
    eval_metric.update(torch.sigmoid(outputs), labels)

    loss += ls.item() * labels.size(0)
    forward_time += end_time - start_time
print(f'test loss: {(loss / len(test_loader.dataset)):.4f}\t'
      f'forward_time: {(1e6 * forward_time / len(test_loader.dataset)):.4f}e-6\t', end='')
eval_metric.compute()
print(f'number_of_parameters: {(sum(param.nelement() for param in model.parameters()) / 1e6):.2f}M')
