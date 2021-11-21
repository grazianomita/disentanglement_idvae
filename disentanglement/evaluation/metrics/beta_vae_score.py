import json
import torch
import numpy as np
from sklearn import linear_model

def log_beta_vae_score(output_file_path, evaluation_time, train_accuracy, eval_accuracy):
    res = {
        'eval_time' : evaluation_time, 
        'train_beta_score' : train_accuracy, 
        'eval_beta_score' : eval_accuracy
    }
    with open(output_file_path, 'w') as fp:
        json.dump(res, fp)

def compute_beta_vae_score(model, ground_truth_data, u_idx, train_dataset_size=10000, eval_dataset_size=5000, batch_size=64, seed=17):
    model.cpu()
    model.eval()
    random_state=np.random.RandomState(seed)
    train_x, train_y = _generate_dataset(model, ground_truth_data, u_idx, train_dataset_size, batch_size, random_state)
    eval_x, eval_y = _generate_dataset(model, ground_truth_data, u_idx, eval_dataset_size, batch_size, random_state)
    l_model = linear_model.LogisticRegression(random_state=random_state)
    l_model.fit(train_x, train_y)
    train_accuracy = l_model.score(train_x, train_y)
    eval_accuracy = l_model.score(eval_x, eval_y)
    return train_accuracy, eval_accuracy

def _generate_dataset(model, ground_truth_data, u_idx, dataset_size, batch_size, random_state):
    inputs = []
    labels = []
    for i in range(dataset_size):
        z, y = _generate_sample(model, ground_truth_data, u_idx, batch_size, random_state)
        inputs.append(z.tolist())
        labels.append(y)
    return inputs, labels

def _generate_sample(model, ground_truth_data, u_idx, batch_size, random_state):
    # Select random coordinate (generative factor) to keep fixed.
    k = random_state.randint(ground_truth_data.num_factors)
    # Sample two mini batches of latent generative factors.
    factors1 = ground_truth_data.sample_factors(batch_size, random_state)
    factors2 = ground_truth_data.sample_factors(batch_size, random_state)
    # Ensure pairs of samples have the same generative factor in common.
    factors2[:, k] = factors1[:, k]
    # Transform latent variables to observation space.
    observations1 = torch.FloatTensor(ground_truth_data.sample_observations_from_factors(factors1, random_state))
    observations2 = torch.FloatTensor(ground_truth_data.sample_observations_from_factors(factors2, random_state))
    b, h, w, c = observations1.shape
    observations1 = observations1.view(b, c, h, w)
    observations2 = observations2.view(b, c, h, w)
    # Compute the latent representation based on the observations.
    z1, mu1, logvar1 = model.encode(observations1, torch.from_numpy(np.array(factors1)/ground_truth_data.factor_sizes).type(torch.FloatTensor)[:, u_idx])
    z2, mu2, logvar2 = model.encode(observations2, torch.from_numpy(np.array(factors2)/ground_truth_data.factor_sizes).type(torch.FloatTensor)[:, u_idx])
    z_diff_mean = torch.mean(torch.abs(z1 - z2), axis=0)
    return z_diff_mean, k