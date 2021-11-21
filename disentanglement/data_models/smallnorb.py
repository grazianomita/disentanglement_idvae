import os
import numpy as np

from disentanglement.data_models.ground_truth_data import GroundTruthData
from disentanglement.data_models.utils import SplitDiscreteStateSpace, StateSpaceAtomIndex
from PIL import Image
from six.moves import range
from six.moves import zip
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class SmallNORB(GroundTruthData, Dataset):
    """
    SmallNORB dataset.
    The data set can be downloaded from
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/. Images are resized to 64x64.
    The ground-truth factors of variation are:
    0 - category (5 different values)
    1 - elevation (9 different values)
    2 - azimuth (18 different values)
    3 - lighting condition (6 different values)
    The instance in each category is randomly sampled when generating the images.
    """
    
    def __init__(self, path="/opt/notebooks/image_datasets/norb", transform=transforms.ToTensor()):
        smallnorb_template = os.path.join(path, "smallnorb-{}-{}.mat")
        smallnorb_chuncks = [ "5x46789x9x18x6x2x96x96-training", "5x01235x9x18x6x2x96x96-testing"]
        self.images, self.features = _load_small_norb_chunks(smallnorb_template, smallnorb_chuncks)
        self.factor_sizes = [5, 10, 9, 18, 6]
        self.full_factor_sizes = self.factor_sizes
        self.latent_factor_indices = [0, 1, 2, 3, 4]
        self.num_total_factors = self.features.shape[1]
        self.index = StateSpaceAtomIndex(self.full_factor_sizes, self.features)
        self.state_space = SplitDiscreteStateSpace(self.full_factor_sizes, self.latent_factor_indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        factors = self.features[idx]/self.full_factor_sizes
        if self.transform:
            sample = sample.reshape(1, 64, 64)
        return sample, factors

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return [64, 64, 1]

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)
    
    def get_dataloader(self, batch_size=128, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def _load_small_norb_chunks(path_template, chunk_names):
    """Loads several chunks of the small norb data set for final use."""
    list_of_images, list_of_features = _load_chunks(path_template, chunk_names)
    features = np.concatenate(list_of_features, axis=0)
    features[:, 3] = features[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
    return np.concatenate(list_of_images, axis=0), features

def _load_chunks(path_template, chunk_names):
    """Loads several chunks of the small norb data set into lists."""
    list_of_images = []
    list_of_features = []
    for chunk_name in chunk_names:
        norb = _read_binary_matrix(path_template.format(chunk_name, "dat"))
        list_of_images.append(_resize_images(norb[:, 0]))
        norb_class = _read_binary_matrix(path_template.format(chunk_name, "cat"))
        norb_info = _read_binary_matrix(path_template.format(chunk_name, "info"))
        list_of_features.append(np.column_stack((norb_class, norb_info)))
    return list_of_images, list_of_features

def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    with open(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])
        dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double"
        }
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data

def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images / 255.