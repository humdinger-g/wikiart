import os
import unicodedata
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from PIL import Image

from typing import List, Dict, Tuple, Optional, Any

class WikiartDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 paths: List, 
                 artist2idx: Dict[str, int], 
                 style2idx: Dict[str, int], 
                 transforms: torchvision.transforms.Compose) -> None:
        
        self.root = root
        self.transform = transforms
        self.paths = paths
        self.artist2idx = artist2idx
        self.style2idx = style2idx

    def load_painting(self, index: int) -> Image.Image:
        path = self.paths[index]
        return Image.open(path)


    def __len__(self) -> int:
        return len(self.paths)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        try:
            img = self.load_painting(index)
            img = self.transform(img)
            path = os.path.split(self.paths[index])
            style = self.style2idx[path[0].split(os.sep)[-1]]
            artist = self.artist2idx[unicodedata.normalize('NFC', path[1].split('_')[0])]
            return img, style, artist
        except Exception as e:
            print(f'Exception in __getitem__: {e}')
            raise e
        


def get_data(source: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Function that returns lists of filenames, artists and styles from a given directory

    """
    files = []
    artists = []
    styles = set()
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.jpg')) and not filename.startswith('.'):
                files.append(os.path.join(root, filename))
                artists += [filename.split('_')[0]]
                styles.add(root.split(os.sep)[-1])
    return files, artists, styles


def get_paths(source: str, artist2idx: Dict[str, int]) -> List:
    """ 
    Creates list of all paths to images in the given directory

    """
    paths = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.jpg')) and not filename.startswith('.'):
                artist = filename.split(os.sep)[-1].split('_')[0]
                if artist in artist2idx:
                    paths.append(os.path.join(root, filename))
                
    return paths


def create_datasets(root: str,  
                    transforms: torchvision.transforms.Compose,
                    min_paintings: int = 50) -> Tuple[Dataset, Dataset]:
    
    all_files, artists, styles = get_data(root)
    artists_count = Counter(artists)
    artist_count_filtered = {a:c for a, c in artists_count.items()
                            if c >= min_paintings}
    artist2idx, style2idx = dict(), dict()
    for i, artist in enumerate(artist_count_filtered.keys()):
        artist2idx[artist] = i

    for i, style in enumerate(styles):
        style2idx[style] = i
    train_dir, test_dir = os.path.join(root, 'train'), os.path.join(root, 'test')
    train_paths, test_paths = get_paths(train_dir, artist2idx), get_paths(test_dir, artist2idx)

    train_dataset = WikiartDataset(root, train_paths, artist2idx, style2idx, transforms)
    test_dataset = WikiartDataset(root, test_paths, artist2idx, style2idx, transforms)
    return train_dataset, test_dataset


def create_single_dataset(root: str, 
                          transforms: torchvision.transforms.Compose,
                          min_paintings: int = 50,  
                          style2idx: Optional[Dict[str, int]] = None, 
                          artist2idx: Optional[Dict[str, int]] = None) -> Dataset:
    
    all_files, artists, styles = get_data(root)
    
    if not style2idx and not artist2idx:
        artists_count = Counter(artists)
        artist_count_filtered = {a:c for a, c in artists_count.items()
                                if c >= min_paintings}
        artist2idx, style2idx = dict(), dict()
        for i, artist in enumerate(artist_count_filtered.keys()):
            artist2idx[artist] = i

        for i, style in enumerate(styles):
            style2idx[style] = i
    paths = get_paths(root, artist2idx)

    dataset = WikiartDataset(root, paths, artist2idx, style2idx, transforms)
    return dataset


def create_dataloaders(train_dataset: Dataset,
                        test_dataset: Dataset,
                        batch_size: int):

    num_workers = os.cpu_count()
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
    return train_dataloader, test_dataloader