"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 10. The number of classes in the dataset.
    """
    # def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):

    def __init__(self, corrupt_prob=0.0, num_classes=10, num_samples=0, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)
        if num_samples > 0:
            self.choose_samples(num_samples)

    def corrupt_labels(self, corrupt_prob):
        # labels = np.array(
        #     self.train_labels if self.train else self.test_labels)
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        # if self.train:
        #     self.train_labels = labels
        # else:
        #     self.test_labels = labels
        self.targets = labels

    def choose_samples(self, num_samples):
        data = self.data
        labels = self.targets

        values = [(j, data[j], labels[j]) for j in range(len(data))]
        indices_subset = [[v[0] for v in values if v[2] == j][:num_samples]
                          for j in range(self.n_classes)]
        flattened_indices = [i for sub in indices_subset for i in sub]

        self.data = np.stack([values[ind][1]
                              for ind in flattened_indices], axis=0)
        self.targets = [values[ind][2] for ind in flattened_indices]
