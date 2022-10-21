import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Grayscale, Pad, CenterCrop
import nibabel as nib
import SimpleITK as skt
from dataset.preprocess_utils import rescale_intensity, equalize_hist


class Dataloader(Dataset):
    def __init__(
        self, source_dir, target_dir, transform = [ToTensor()],
    ):
        """
        Args:
            source_dir (string): Directory with target the images.
            target_dir (string): Directory with target the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_dir = target_dir
        self.source_dir = source_dir
        self.padding = False
        self.transforms = Compose(transform)
        self.source = os.listdir(source_dir)
        self.target = os.listdir(target_dir)

    @staticmethod
    def load_file(path):
        img = skt.ReadImage(path)
        vol = skt.GetArrayFromImage(img)
        # affine = nii.affine
        vol = rescale_intensity(vol)
        vol = equalize_hist(vol)
        # vol = (vol - vol.min()) / (vol.max() - vol.min())
        return vol

    def __len__(self) -> int:
        """
        Returns the length of the Dataloader class
        :return: integer with length of Dataloader class
        """
        return len(self.source)

    def __getitem__(self, idx):
        """
        Returns items from the Dataloader class in interpretable format
        :param idx: Index of instance that has to be accessed from class
        :return: Tuple of numpy arrays containing source and target image
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        source = self.load_file(f"{self.source_dir}/{self.source[idx]}")
        target = self.load_file(f"{self.target_dir}/{self.target[idx]}")
        source = torch.as_tensor(self.transforms(source), dtype=torch.float)
        target = torch.as_tensor(self.transforms(target), dtype=torch.float)

        if self.padding:
            mask = torch.ones_like(target)
            x = torch.arange(source.shape[0])
            mask[x*2, :, :] = source
            source = mask

        return source, target