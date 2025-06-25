import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class MyCrossover(nn.Module):

    def __init__(self, f_prob=0):
        super(MyCrossover, self).__init__()
        self.f_prob = f_prob  #prob of sampling 1. for Bern
        self.mm = torch.distributions.bernoulli.Bernoulli(torch.tensor([self.f_prob], device='cuda'))

    def forward(self, h, h_in):
        if self.training:

            crossover_mask = self.mm.sample(h.shape).squeeze(-1)
            #pdb.set_trace()
            h = h_in * crossover_mask + h * (1 - crossover_mask)
            #pdb.set_trace()

        else:
            h = h_in * self.f_prob + h * (1- self.f_prob)

        return h



class MyMutation(nn.BatchNorm1d):
    def __init__(self, num_features, mutate_prob, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyMutation, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.mutate_prob = mutate_prob

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0])
            # use biased var in train
            var = input.var([0], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        mean = self.running_mean
        var = self.running_var

        #prob_mutate = 0.5
        #if self.training:
        #    mm = torch.distributions.bernoulli.Bernoulli(torch.tensor([prob_mutate], device='cuda'))
        #    mutate_mask = mm.sample(x.shape).squeeze(-1)
        #    #pdb.set_trace()
        #    x = x * mutate_mask + x_in * (1 - mutate_mask)
        #    #pdb.set_trace()

        #else:
        #    x = x * prob_mutate + x_in * (1- prob_mutate)
        #    pass
        gaussion_noise = torch.randn(input.shape).type_as(input)

        prob_mutate = self.mutate_prob
        if self.training:
            mm = torch.distributions.bernoulli.Bernoulli(torch.tensor([prob_mutate], device='cuda'))
            mutate_mask = mm.sample(input.shape).squeeze(-1)
            input = (gaussion_noise * var + mean)*mutate_mask + input * (1 - mutate_mask)
            pass
        else:

            input = mean * prob_mutate + input * (1 - prob_mutate)
            pass

        #pdb.set_trace()
        #input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        #if self.affine:
        #    input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input
