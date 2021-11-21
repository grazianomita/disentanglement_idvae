import json
import numpy as np
from disentanglement.evaluation.metrics import utils

def log_mig(output_file_path, evaluation_time, mig_score):
    res = {
        'eval_time' : evaluation_time, 
        'mig_score' : mig_score
    }
    with open(output_file_path, 'w') as fp:
        json.dump(res, fp)

def compute_mig(model, ground_truth_data, u_idx, num_train=10000, batch_size=16, seed=17):
    """Computes the mutual information gap."""
    model.cpu()
    model.eval()
    random_state=np.random.RandomState(seed)
    mus_train, ys_train = utils.generate_batch_factor_code(model, ground_truth_data, u_idx, num_train, random_state, batch_size)
    assert mus_train.shape[1] == num_train
    return _compute_mig(mus_train, ys_train)

def _compute_mig(mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    discretized_mus = utils.make_discretizer(mus_train)
    m = utils.discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    # m is [num_latents, num_factors]
    entropy = utils.discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    return np.mean([res for res in np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]) if res < np.inf])