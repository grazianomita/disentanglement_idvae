import os
import h5py
import numpy as np

from disentanglement.data_models.ground_truth_data import GroundTruthData
from disentanglement.data_models.utils import SplitDiscreteStateSpace
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class Shapes3D(GroundTruthData, Dataset):
    """
    Shapes3D dataset.
    The data set was originally introduced in "Disentangling by Factorising".
    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    """

    def __init__(
        self,
        path="/opt/notebooks/generative_models/datasets/shapes3d/3dshapes.h5",
        transform=transforms.ToTensor()
    ):
        data = h5py.File(path, 'r')
        images = data['images'][()]
        labels = data['labels'][()]
        n_samples = images.shape[0]
        self.images = (images.reshape([n_samples, 64, 64, 3]).astype(np.float32) / 255.)
        features = labels.reshape([n_samples, 6])
        self.factor_sizes = [10, 10, 10, 8, 4, 15]
		self.full_factor_sizes = self.factor_sizes
        self.latent_factor_indices = list(range(6))
        self.num_total_factors = features.shape[1]
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        factors = self.__get_factors_from_image_idx(idx)
        if self.transform:
            sample = self.transform(sample)
        return sample, factors
    
    def __get_factors_from_image_idx(self, idx):
        num_factors = len(self.factor_sizes)
        f = np.zeros(shape=(num_factors))
        for i in range(num_factors):
            q = int(idx/self.factor_sizes[num_factors-i-1])
            r = idx%self.factor_sizes[num_factors-i-1]
            f[num_factors-i-1] = r
            idx = q
        return f/self.factor_sizes

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [64, 64, 3]

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices]
    
    def get_dataloader(self, batch_size=128, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
