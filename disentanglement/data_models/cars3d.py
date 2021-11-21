import os
import PIL
import numpy as np
import scipy.io as sio

from disentanglement.data_models.ground_truth_data import GroundTruthData
from disentanglement.data_models.utils import SplitDiscreteStateSpace, StateSpaceAtomIndex
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.extmath import cartesian

class Cars3D(GroundTruthData, Dataset):
    """
    Cars3D data set.
    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.
    The ground-truth factors of variation are:
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)
    """

    def __init__(
        self,
        path="/opt/notebooks/generative_models/datasets/cars3d/", 
        transform=transforms.ToTensor()
    ):
        self.factor_sizes = [4, 24, 183]
        self.full_factor_sizes = self.factor_sizes
        features = cartesian([np.array(list(range(i))) for i in self.factor_sizes])
        self.latent_factor_indices = [0, 1, 2]
        self.num_total_factors = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
        self.data_shape = [64, 64, 3]
        self.images = self._load_data(path)
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
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return self.images[indices].astype(np.float32)

    def _load_data(self, path):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        all_files = [x for x in os.listdir(path) if ".mat" in x]
        for i, filename in enumerate(all_files):
            data_mesh = self._load_mesh(path, filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i, len(factor1) * len(factor2))
            ])
            indexes = self.index.features_to_index(all_factors)
            dataset[indexes] = data_mesh
        return dataset

    def _load_mesh(self, path, filename):
        """Parses a single source file and rescales contained images."""
        with open(os.path.join(path, filename), "rb") as f:
            mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
        for i in range(flattened_mesh.shape[0]):
            pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
            pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh * 1. / 255

    def get_dataloader(self, batch_size=128, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
