#! -*- coding: utf-8
import os
import os.path as path
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import roc_curve
from torchvision import transforms

from losses import AUC

try:
    if os.name == "POSIX":
        import matplotlib
        matplotlib.use("Agg")
finally:
    from matplotlib import pyplot as plt


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = self.conv_layer()
        self.linear = self.linear_layer()

    def forward(self, x):
        x = self.conv(x).view(-1, 12*12*64)
        return self.linear(x)

    def conv_layer(self):
        layer = OrderedDict()
        layer["conv1"] = torch.nn.Conv2d(1, 32, (3, 3))  # 28x28x32 -> 26x26x32
        layer["conv2"] = torch.nn.Conv2d(
            32, 64, (3, 3))  # 26x26x32 -> 24x24x64
        layer["pool2"] = torch.nn.MaxPool2d(2, 2)  # 12x12x64
        layer["dropout2"] = torch.nn.Dropout2d()

        return torch.nn.Sequential(layer)

    def linear_layer(self):
        layer = OrderedDict()
        layer["linear1"] = torch.nn.Linear(12*12*64, 128)
        layer["relu1"] = torch.nn.ReLU()
        layer["dropout1"] = torch.nn.Dropout2d()
        layer["linear2"] = torch.nn.Linear(128, 1)

        return torch.nn.Sequential(layer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("outdir", type=str)
    parser.add_argument("--target", type=int, nargs=2, default=[6, 8])
    parser.add_argument("--sample_rate", type=float,
                        nargs=2, default=[1.0, 1.0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cuda", default=False, action="store_true")
    args = parser.parse_args()

    assert len(set(args.target)) == 2

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if args.cuda else "cpu"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform)
    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform)

    # shape: 1x28x28
    train_x, train_label = [], []
    test_x, test_label = [], []

    for vals, labels, dataset in zip([train_x, test_x],
                                     [train_label, test_label],
                                     [trainset, testset]):
        for v, l in dataset:
            if not l in args.target:
                continue
            vals.append(v)
            labels.append(l)
    train_x = torch.stack(train_x, dim=0)
    test_x = torch.stack(test_x, dim=0)
    train_label, test_label = np.array(train_label), np.array(test_label)
    print(train_x.shape, len(train_label), test_x.shape, len(test_label))

    model = Model()
    criterion = AUC()
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)
    train_x = train_x.to(device)
    train_y = torch.from_numpy(
        (train_label == args.target[1]).astype(float)).view(-1, 1).to(device)
    test_x = test_x.to(device)
    test_y = torch.from_numpy(
        (test_label == args.target[1]).astype(float)).view(-1, 1).to(device)
    niter = (len(train_x) + args.batch_size - 1) // args.batch_size
    model.train()
    for epoch in range(1, args.epochs+1):
        o = model(train_x)
        loss = criterion(o, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("%d %f" % (epoch, loss.detach().cpu().item()))

    os.makedirs(args.outdir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for tag, x, label in zip(["train", "test"], [train_x, test_x], [train_label, test_label]):
            o = model(x).detach().cpu().numpy().ravel()
            df = pd.DataFrame({"label": label.ravel(), "value": o})
            df.to_csv(path.join(args.outdir, "%s.csv" % (tag)), index=False)
            np.save(path.join(args.outdir, tag),
                    x.detach().cpu().squeeze().numpy())
            pass
        pass

    for tag in ["train", "test"]:
        df = pd.read_csv(path.join(args.outdir, "%s.csv" % tag))
        fpr, tpr, thresholds = roc_curve(
            df.label.values, df.value.values, pos_label=args.target[-1])
        threshold = thresholds[np.argmax(1/(0.5 * (1/tpr + 1/(1-fpr))))]
        print(tag, threshold)
        df["sertainty"] = np.abs(df.value.values - threshold)
        df.to_csv(path.join(args.outdir, "%s.csv" % tag), index=False)

        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax = ax.ravel()
        target0 = df.loc[df.label == args.target[0], "value"].values
        target1 = df.loc[df.label == args.target[1], "value"].values
        ax[0].hist(target0, bins=100, label=str(args.target[0]))
        ax[0].legend()
        ax[1].hist(target1, bins=100, label=str(args.target[1]))
        ax[1].legend()
        plt.savefig(path.join(args.outdir, "hist_%s.png" % (tag)))
        plt.close()

        x = np.load(path.join(args.outdir, "%s.npy" % tag))

        df.sort_values(by="sertainty", inplace=True, ascending=True)
        row, col = 40,40

        vals=[]
        for r in range(row):
            rowvals = []
            for c in range(col):
                i = r * row + c
                rowvals.append(x[df.iloc[i].name])
            vals.append(np.concatenate(rowvals, axis=1))
        vals = np.concatenate(vals, axis=0)
        plt.imshow(vals)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path.join(args.outdir, "dirties_%s.png" % (tag)))
        plt.close()

        # fig, ax = plt.subplots(40, 40,
        #                        sharex=True,
        #                        sharey=True,
        #                        figsize=(10, 10))
        # ax = ax.ravel()
        # for i, a in enumerate(ax):
        #     v = x[df.iloc[i].name]
        #     a.imshow(v)
        #     a.axis("off")
        # plt.tight_layout()
        # plt.savefig(path.join(args.outdir, "dirties_%s.png" % (tag)))
        # plt.close()

        df = df.iloc[:row*col]
        dirties = x[df.index]
        df.to_csv(path.join(args.outdir, "dirty_%s.csv" % (tag)), index=False)
        np.save(path.join(args.outdir, "dirty_%s" % (tag)), dirties)
