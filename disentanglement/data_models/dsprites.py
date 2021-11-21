import os
import numpy as np

from disentanglement.data_models.ground_truth_data import GroundTruthData
from disentanglement.data_models.utils import SplitDiscreteStateSpace
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class DSprites(GroundTruthData, Dataset):
    """
    DSprites dataset.
    The data set was originally introduced in "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework" and can be downloaded from
    https://github.com/deepmind/dsprites-dataset.
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    In reality we have 6 factors. The first one is 'color' but it is always black.
    """

    def __init__(
        self, 
        path="/opt/notebooks/generative_models/datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", 
        latent_factor_indices=None,
        transform=transforms.ToTensor()
    ):
        # By default, all factors (including shape) are considered ground truth factors.
        if latent_factor_indices is None:
            latent_factor_indices = list(range(6))
        self.latent_factor_indices = latent_factor_indices
        self.data_shape = [64, 64, 1]
        # Load the data so that we can sample from it.
        data = np.load(path, encoding="latin1", allow_pickle=True)
        self.images = np.array(data["imgs"], dtype=np.float32)
        self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)
        self.full_factor_sizes = [1, 3, 6, 40, 32, 32]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        factors = self.__get_factors_from_image_idx(idx)
        # sample_max = sample.max()
        # sample = sample * 255 / sample_max
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        # sample = sample.reshape(sample.shape + (1,))
        if self.transform:
            sample = sample.reshape(1, 64, 64)
            # sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, factors
    
    def __get_factors_from_image_idx(self, idx):
        num_factors = len(self.full_factor_sizes)
        f = np.zeros(shape=(num_factors))
        for i in range(num_factors):
            q = int(idx/self.full_factor_sizes[num_factors-i-1])
            r = idx%self.full_factor_sizes[num_factors-i-1]
            f[num_factors-i-1] = r
            idx = q
        return f/self.full_factor_sizes

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        return self.sample_observations_from_factors_no_color(factors, random_state)

    def sample_observations_from_factors_no_color(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)
    
    def get_dataloader(self, batch_size=128, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
