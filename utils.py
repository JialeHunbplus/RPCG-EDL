import os
import random
import numpy as np
import torch
import torch.nn as nn
import dgl
import logging
from collections import Counter
import math
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10,
    "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
    "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}

CHARPROTLEN = 25

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 核心接口：支持底物三维特征打包
def prottrans_graph_collate_func(x):
    d, p, y, w = zip(*x)
    molt5_batch = torch.stack([item[0] for item in d])
    maccs_batch = torch.stack([item[1] for item in d])
    pc_batch = torch.stack([item[2] for item in d])
    frag_graph_batch = dgl.batch([item[3] for item in d])
    # 打包交付给 models.py 的四模态门控
    d_batch = (molt5_batch, maccs_batch, pc_batch, frag_graph_batch) 
    
    if isinstance(p[0], tuple):
        protein_seq_batch = torch.stack([item[0] for item in p])
        protein_struct_batch = torch.stack([item[1] for item in p])
        p_batch = (protein_seq_batch, protein_struct_batch)
    else:
        p_batch = torch.tensor(np.array(p))
    w = np.array(w)
    return d_batch, p_batch, torch.tensor(y), torch.tensor(w)

def integer_graph_collate_func(x):
    d, p, y, w = zip(*x)
    d = dgl.batch(d)
    w = np.array(w)
    return d, torch.tensor(np.array(p)), torch.tensor(y), torch.tensor(w)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

def integer_label_protein(sequence, max_length=1200):
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as padding."
            )
    return encoding

def evidential_loss(mu, v, alpha, beta, targets, lamba = 0.1):
    def Gamma(x):
        return torch.exp(torch.lgamma(x))

    coeff_denom = 4 * Gamma(alpha) * v * torch.sqrt(beta)
    coeff_num = Gamma(alpha - 0.5)
    coeff = coeff_num / coeff_denom

    second_term = 2 * beta * (1 + v)
    second_term += (2 * alpha - 1) * v * torch.pow((targets - mu), 2)
    L_SOS = coeff * second_term

    L_REG = torch.pow((targets - mu), 2) * (2 * alpha + v)
    loss_val = L_SOS + lamba * L_REG

    return loss_val

def Smooth_Label_CBW(Label_new,beta=0.9):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label*10) for label in labels]
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    eff_label_dist = []
    for i in range(len(emp_label_dist)):
        eff_label_dist.append((1-math.pow(beta, emp_label_dist[i])) / (1-beta))
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    return weights

def Smooth_Label_CSW(Label_new):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label * 10) for label in labels]
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    eff_label_dist = emp_label_dist
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    return weights

def Smooth_Label_DMW(Label_new):
    labels = Label_new
    weights = np.ones([len(Label_new)], dtype=float)
    for i in range(len(labels)):
        if Label_new[i] > 5 or Label_new[i] < -5:
            weights[i] = 2
    return weights

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window

def Smooth_Label_LDS(Label_new):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label*4) for label in labels]
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=3, sigma=1)
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    return weights

class EvidentialLossSumOfSquares(nn.Module):
    def __init__(self, debug=False, return_all=False):
        super(EvidentialLossSumOfSquares, self).__init__()
        self.debug = debug
        self.return_all_values = return_all
        self.MAX_CLAMP_VALUE = 5.0

    def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
        alpha_2 = torch.ones_like(mu1) * 1.0
        beta_2 = torch.ones_like(mu1) * 0.1
        lambda_2 = torch.ones_like(mu1) * 1.0

        t1 = 0.5 * (alpha_1 / beta_1) * ((mu1 - mu2) ** 2) * lambda_2
        t2 = 0.5 * lambda_2 / lambda_1
        t3 = alpha_2 * torch.log(beta_1 / beta_2)
        t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
        t5 = (alpha_1 - alpha_2) * torch.digamma(alpha_1)
        t6 = -(beta_1 - beta_2) * (alpha_1 / beta_1)
        return (t1 + t2 - 0.5 + t3 + t4 + t5 + t6)

    def forward(self, inputs, targets):
        assert torch.is_tensor(inputs)
        assert torch.is_tensor(targets)
        assert (inputs[:, 1] > 0).all()
        assert (inputs[:, 2] > 0).all()
        assert (inputs[:, 3] > 0).all()

        targets = targets.view(-1)
        y = inputs[:, 0].view(-1)
        a = inputs[:, 1].view(-1) + 1.0
        b = inputs[:, 2].view(-1) + 0.1
        l = inputs[:, 3].view(-1) + 1.0

        J1 = torch.lgamma(a - 0.5)
        J2 = -torch.log(torch.tensor([4.0]))
        J3 = -torch.lgamma(a)
        J4 = -torch.log(l)
        J5 = -0.5 * torch.log(b)
        J6 = torch.log(2 * b * (1 + l) + (2 * a - 1) * l * (y - targets) ** 2)

        J = J1 + J2 + J3 + J4 + J5 + J6
        Kl_divergence = self.kl_divergence_nig(y, targets, a, b, l)

        loss = torch.exp(J) + Kl_divergence
        if self.return_all_values:
            ret_loss = loss
        else:
            ret_loss = loss.mean()
        return ret_loss

class EvidentialLossNLL(nn.Module):
    def __init__(self, debug=False, return_all=False):
        super(EvidentialLossNLL, self).__init__()
        self.debug = debug
        self.return_all_values = return_all
        self.MAX_CLAMP_VALUE = 5.0

    def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
        alpha_2 = torch.ones_like(mu1) * 1.0
        beta_2 = torch.ones_like(mu1) * 0.1
        lambda_2 = torch.ones_like(mu1) * 1.0

        t1 = 0.5 * (alpha_1 / beta_1) * ((mu1 - mu2) ** 2) * lambda_2
        t2 = 0.5 * lambda_2 / lambda_1
        t3 = alpha_2 * torch.log(beta_1 / beta_2)
        t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
        t5 = (alpha_1 - alpha_2) * torch.digamma(alpha_1)
        t6 = -(beta_1 - beta_2) * (alpha_1 / beta_1)
        return (t1 + t2 - 0.5 + t3 + t4 + t5 + t6)

    def forward(self, mu, v, alpha, beta, targets, lam, epsilon):
        assert torch.is_tensor(targets)
        targets = targets.view(-1)

        twoBlambda = 2 * beta * (1 + v)
        nll = 0.5 * torch.log(np.pi / v) \
              - alpha * torch.log(twoBlambda) \
              + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)
        L_NLL = nll
        error = torch.abs((targets - mu))
        reg = error * (2 * v + alpha)
        L_REG = reg
        loss = L_NLL + lam * (L_REG - epsilon)
        ret_loss = loss.mean()
        return ret_loss
