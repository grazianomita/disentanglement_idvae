import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from disentanglement.evaluation.metrics import utils

def log_modularity(output_file_path, evaluation_time, modularity, explicitness_score_train, explicitness_score_test):
    res = {
        'eval_time' : evaluation_time, 
        'modularity' : modularity,
        'explicitness_score_train': explicitness_score_train,
        'explicitness_score_test': explicitness_score_test
    }
    with open(output_file_path, 'w') as fp:
        json.dump(res, fp)

def compute_modularity_explicitness(model, ground_truth_data, u_idx, num_train=10000, num_test=5000, batch_size=16, seed=17):
    model.cpu()
    model.eval()
    random_state=np.random.RandomState(seed)
    mus_train, ys_train = utils.generate_batch_factor_code(model, ground_truth_data, u_idx, num_train, random_state, batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(model, ground_truth_data, u_idx, num_test, random_state, batch_size)
    discretized_mus = utils.make_discretizer(mus_train)
    mutual_information = utils.discrete_mutual_info(discretized_mus, ys_train)
    # Mutual information should have shape [num_codes, num_factors].
    assert mutual_information.shape[0] == mus_train.shape[0]
    assert mutual_information.shape[1] == ys_train.shape[0]
    modularity_score = modularity(mutual_information)
    explicitness_score_train = np.zeros([ys_train.shape[0], 1])
    explicitness_score_test = np.zeros([ys_test.shape[0], 1])
    mus_train_norm, mean_mus, stddev_mus = utils.normalize_data(mus_train)
    mus_test_norm, _, _ = utils.normalize_data(mus_test, mean_mus, stddev_mus)
    for i in range(ys_train.shape[0]):
        explicitness_score_train[i], explicitness_score_test[i] = explicitness_per_factor(mus_train_norm, ys_train[i, :], mus_test_norm, ys_test[i, :])
    explicitness_score_train = np.mean(explicitness_score_train)
    explicitness_score_test = np.mean(explicitness_score_test)
    return modularity_score, explicitness_score_train, explicitness_score_test

def explicitness_per_factor(mus_train, y_train, mus_test, y_test):
    """Compute explicitness score for a factor as ROC-AUC of a classifier."""
    x_train = np.transpose(mus_train)
    x_test = np.transpose(mus_test)
    clf = LogisticRegression().fit(x_train, y_train)
    y_pred_train = clf.predict_proba(x_train)
    y_pred_test = clf.predict_proba(x_test)
    mlb = MultiLabelBinarizer()
    roc_train = roc_auc_score(mlb.fit_transform(np.expand_dims(y_train, 1)), y_pred_train)
    roc_test = roc_auc_score(mlb.fit_transform(np.expand_dims(y_test, 1)), y_pred_test)
    return roc_train, roc_test

def modularity(mutual_information):
    """Computes the modularity from mutual information."""
    # Mutual information has shape [num_codes, num_factors].
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] -1.)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)