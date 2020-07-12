# This code computes lower bounds on adversarial risk for CIFAR10 dataset and Gaussian mixtures based on them.
# The adversarial model is an L infinity norm perturbation adversary with budget epsilon.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

import numpy as np
from lapsolver import solve_dense


from matplotlib import pyplot as plt
import pandas as pd

from scipy.stats import norm 
from scipy.stats import ncx2
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist


# Select the training dataset of CIFAR10
x_train = datasets.CIFAR10("../data", train=True, download=True, transform=transforms.ToTensor())

# Filtering for class labels 3 and 5 (corresponds to cats and dogs)
target_inds_train = [i for i, j in enumerate(x_train.targets) if (j==3)|(j==5)]
x_train.data = x_train.data[target_inds_train]
x_train.targets = [x_train.targets[i] for i in target_inds_train]

train_loader = DataLoader(x_train, batch_size = 100, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

n = int(len(x_train.data)/2)
dim = np.size(x_train.data[0])

x0_inds = [i for i, j in enumerate(x_train.targets) if (j==3)]
x1_inds = [i for i, j in enumerate(x_train.targets) if (j==5)]

x0 = x_train.data[x0_inds].reshape((n,-1)).astype(float)
x1 = x_train.data[x1_inds].reshape((n,-1)).astype(float)


dist0 = pdist(x0, 'chebyshev')
dist1 = pdist(x1, 'chebyshev')

# Choose a sigma^* value based on the mean distance between pairs of datapoints from the same class
sig_opt = (np.mean(dist0)+ np.mean(dist1))/4

dists = cdist(x0,x1,'chebyshev')

sig_range = [sig_opt/3, sig_opt, sig_opt*3]

# Select a range of adversarial budgets
ep_range = np.arange(0, 111, 5)

err_lb_arr = np.zeros(len(ep_range))
err_lb_arr_sig = np.zeros((len(sig_range), len(ep_range)))

for epid in range(len(ep_range)):
    epsilon = ep_range[epid]
    costs = np.ones_like(dists)
    costs[dists<=2*epsilon]=0
    rids, cids = solve_dense(costs)
    opt_cost = 0;
    for r,c in zip(rids, cids):
        opt_cost = opt_cost + costs[r,c] 
    err_lb = 0.5*(1-opt_cost/n)
    err_lb_arr[epid] = err_lb
    print('Error lower bound for epsilon = ' + str(epsilon) + ' is '+ str(err_lb)+'.')

    for sigid in range(len(sig_range)):
        sig = sig_range[sigid]
        opt_cost_sig = 0
        for r,c in zip(rids, cids): 
            mean_diff = dists[r,c]
            prob_temp = 2*norm.cdf((mean_diff/2-epsilon)/sig)-1   
            prob_rc = costs[r,c]*prob_temp
            opt_cost_sig = opt_cost_sig + prob_rc        
        err_lb_sig = 0.5*(1-opt_cost_sig/n)
        err_lb_arr_sig[sigid,epid] = err_lb_sig
        print('(With sigma) Error lower bound for epsilon = ' + str(epsilon) + ' and sigma = ' + str(sig) +  ' is '+ str(err_lb_sig)+'.')


plt.figure()
plt.plot(ep_range, err_lb_arr, 'o-', label=r'$\sigma$ = 0', color='k')
plt.plot(ep_range, err_lb_arr_sig[0], 'o-', label=r'$\sigma = \sigma^*/3$', color='purple')
plt.plot(ep_range, err_lb_arr_sig[1], 'o-', label=r'$\sigma = \sigma^*$', color='mediumorchid')
plt.plot(ep_range, err_lb_arr_sig[2], 'o-', label=r'$\sigma = 3 \sigma^*$', color='deeppink')

plt.legend(fontsize = 'xx-large')
plt.xlabel(r'Adversarial budget, $\epsilon\times 255$', fontsize = 'xx-large')
plt.ylabel('Adversarial error lower bound', fontsize = 'xx-large')
plt.grid()
plt.show()
plt.savefig('cifar_linf.png', bbox_inches='tight')







