import os
import re
import pydoc
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)


def rename_layers(
    state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]
) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def rle2mask(rle: str, height: int = 256, width: int = 1600) -> np.array:
    rows, cols = height, width
    rle_nums = [int(numstring) for numstring in rle.split(" ")]
    rle_pairs = np.array(rle_nums).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index : index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    return img


def combine_masks(row: pd.DataFrame, height: int = 256, width: int = 1600) -> np.array:
    rle1 = row["1"]
    rle2 = row["2"]
    rle3 = row["3"]
    rle4 = row["4"]

    mask1 = rle2mask(rle1, height, width)
    mask2 = rle2mask(rle2, height, width)
    mask3 = rle2mask(rle3, height, width)
    mask4 = rle2mask(rle4, height, width)

    combined = 255 * (mask1 + mask2 + mask3 + mask4)
    combined = combined.clip(0, 255).astype("uint8")
    return combined


def save_masks(
    df: pd.DataFrame, dest_dir: str, height: int = 256, width: int = 1600
) -> None:
    for _, row in df.iterrows():
        fname = row["fname"]
        combined_mask = combine_masks(row, height, width)
        cv2.imwrite(os.path.join(dest_dir, fname), 255 * combined_mask)


def copy_images(df: pd.DataFrame, source_dir: str, dest_dir: str) -> None:
    for _, row in df.iterrows():
        source_fname = os.path.join(source_dir, row["fname"])
        dest_fname = os.path.join(dest_dir, row["fname"])
        shutil.copy(source_fname, dest_fname)
