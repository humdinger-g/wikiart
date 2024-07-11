import os
from tqdm.autonotebook import tqdm
import random
import shutil
import numpy as np

from typing import List, Union

def split(arr: Union[List, np.array],
          k: int = 2) -> List:
    """
    helper function: creates a partition of an array into k arrays, deleting extra elements
    example: split([1,2,3,4,5,6,7], 2) -> [[1,2,3],[4,5,6]]
    
    """
    partition = []
    indices = [len(arr) // k * i for i in range(k+1)]
    for i in range(k):
        a, b = indices[i:i+2]
        partition += [arr[a:b]]
    return partition


def split_dataset(wikiart_path: str, 
                  partition_path: str, 
                  k: int = 2,
                  create_only: int = 1,
                  test_size: float = 0.1, 
                  zip: bool = False, 
                  delete: bool = False) -> None:
    """
    Parameters
    ----------
    wikiart_path : str
        path to the original dataset
    partition_path : str
        path to which partition will be saved
    k : int
        number of parts in partition; each will have size of ~34/k Gb
    create_only : int
        number of parts out of k to create. For example, if k=5 and create_only=2,
        only 2/5 of the data will be loaded
    test_size : float (0, 1)
        fraction of data that will be stored as test data
    zip: bool, default=False
        if True, created parts will be .zip
    delete: bool, defalt=False
        if True, deletes original dataset after partition
        (it's done sequentially, so additional memory required is ~34/k Gb)

    """

    styles = []
    for item in os.listdir(wikiart_path):
        item_path = os.path.join(wikiart_path, item)
        if os.path.isdir(item_path):
            styles += [item]

    for i in range(k):
        if i > create_only:
            break
        for style in styles:
            os.makedirs(os.path.join(partition_path, f'part{i+1}', 'train', style))
            os.makedirs(os.path.join(partition_path, f'part{i+1}', 'test', style))

    for style in tqdm(os.listdir(wikiart_path)):
        style_path = os.path.join(wikiart_path, style)
        if os.path.isdir(style_path):
            all_paintings = os.listdir(style_path)
            random.shuffle(all_paintings)
            partitions = split(all_paintings, k)
            for i in range(k):
                if i > create_only:
                    break
                train = partitions[i][:int(len(partitions[i])*(1-test_size))]
                test = partitions[i][int(len(partitions[i])*(1-test_size)):]
                for painting in train:
                    from_path = os.path.join(wikiart_path, style, painting)
                    to_path = os.path.join(partition_path, f'part{i+1}', 'train', style, painting)
                    shutil.copy2(from_path, to_path)
                    if delete:
                        os.remove(from_path)
                for painting in test:
                    from_path = os.path.join(wikiart_path, style, painting)
                    to_path = os.path.join(partition_path, f'part{i+1}', 'test', style, painting)
                    shutil.copy2(from_path, to_path)
                    if delete:
                        os.remove(from_path)
    if zip:
        for i in tqdm(range(1, k+1)):
            folder_name = f'part{i}'
            folder_path = os.path.join(partition_path, folder_name)
            zip_file_name = f'{folder_name}.zip'
            zip_file_path = os.path.join(partition_path, zip_file_name)

            shutil.make_archive(zip_file_path[:-4], 'zip', folder_path)

            shutil.rmtree(folder_path)


def copy_folder(k: int,
                path1: str, 
                path2: str) -> None:
    
    source_folder = os.path.join(path1, f"part_{k}")
    dest_folder = os.path.join(path2, f"part_{k}")
    del_folder = os.path.join(path2, f"part_{k-1}")

    if os.path.exists(del_folder):
        shutil.rmtree(del_folder)

    shutil.copytree(source_folder, dest_folder)