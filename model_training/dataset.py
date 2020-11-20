import re
from PIL import Image
import torch
from torch.utils.data import Dataset


class BookcoverDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.transforms = transforms

    def get_target_from_path(self, path):
        target_pattern = re.compile(r"[0-9.]+(?=\.[a-z]+)")
        target = float(re.search(target_pattern, path).group())
        return torch.tensor([target])

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        y = self.get_target_from_path(self.image_paths[index])

        if self.transforms:
            x = self.transforms(x)

        return x, y

    def __len__(self):
        return len(self.image_paths)
