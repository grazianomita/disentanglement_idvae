import torch
import numpy as np
import sklearn
from sklearn.metrics import mutual_info_score

def generate_batch_factor_code(model, ground_truth_data, u_idx, num_points, random_state, batch_size):
    """Sample a single training sample based on a mini-batch of ground-truth data."""
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = ground_truth_data.sample(num_points_iter, random_state)
        current_observations = torch.FloatTensor(current_observations)
        b, h, w, c = current_observations.shape
        current_observations = current_observations.view(b, c, h, w)
        #current_observations = torch.from_numpy(current_observations).float()
        current_factors = torch.from_numpy(current_factors).float()
        if i == 0:
            factors = current_factors[:, u_idx]
            # representations = representation_function(current_observations)
            representations, _, _ = model.encode(current_observations, torch.from_numpy(np.array(current_factors)/ground_truth_data.full_factor_sizes).type(torch.FloatTensor)[:, u_idx])
            representations = representations.detach().numpy()
        else:
            factors = np.vstack((factors, current_factors[:, u_idx]))
            r, _, _ = model.encode(current_observations, torch.from_numpy(np.array(current_factors)/ground_truth_data.full_factor_sizes).type(torch.FloatTensor)[:, u_idx])
            r = r.detach().numpy()
            representations = np.vstack((representations, r))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)


def make_discretizer(target, num_bins=20,):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[j, :], ys[j, :])
    return h


def normalize_data(data, mean=None, stddev=None):
    if mean is None:
        mean = np.mean(data, axis=1)
    if stddev is None:
        stddev = np.std(data, axis=1)
    return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev
