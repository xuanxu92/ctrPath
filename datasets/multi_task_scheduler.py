"""
Adapted from https://github.com/salesforce/UniControl/blob/main/train_util/multi_task_scheduler.py

The License of UniControl is as follows:
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
"""

import math
import numpy as np

from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DistributedSampler


class BatchSchedulerSampler(Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, distributed: bool = True, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.distributed = distributed
        self.shuffle = shuffle

        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if not self.distributed:
                if self.shuffle:
                    sampler = RandomSampler(cur_dataset)
                else:
                    sampler = SequentialSampler(cur_dataset)
            else:
                sampler = DistributedSampler(cur_dataset, shuffle=self.shuffle)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            if self.shuffle:
                perm = np.random.permutation(self.number_of_datasets)
            else:
                perm = np.arange(self.number_of_datasets)
            for i in perm:
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):  # batch with one task/dataset
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
