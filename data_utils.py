import os
import shutil
from typing import Tuple
import cv2
import numpy as np
import pandas as pd


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
