import glob
import json
import os
from re import S
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import skimage
from sympy import N
from networkx import non_randomness
from PIL import Image
from scipy.ndimage import rotate
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
from skimage import io as skio

from skimage.measure import label, regionprops, regionprops_table
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import tracemalloc
import cv2
from sys import platform
from tifffile import TiffFile, tifffile
import polars as pl
import tqdm


class CellIdentifier():
    """
    A class to represent a cell.

    Attributes:
        proj_name (str): The name of the project.
        fov (str): The name of the field of view.
        cell_id (int): The unique identifier of the cell.
        channel (str): The channel (marker) used for the cell imaging.
        label (int): The label for the cell.
    """

    def __init__(self, proj_name:str, fov:str, cell_id:int, channel:str, label:int=-1):
        self.proj_name = proj_name
        self.fov = fov
        self.cell_id = cell_id
        self.channel = channel
        self.label = label

    def to_dict(self):
        return {
            'proj_name': self.proj_name,
            'fov': self.fov,
            'cell_id': self.cell_id,
            'channel': self.channel,
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['proj_name'], data['fov'], data['cell_id'], data['channel'], data['label'])


    def __str__(self):
        return '_'.join([self.proj_name, self.fov, self.channel, 'cell', str(self.cell_id)])


    def __eq__(self, other):
        return (self.proj_name == other.proj_name) and (self.fov == other.fov) and (self.cell_id == other.cell_id) and (self.channel == other.channel) 


    def __hash__(self):
        return hash(str(self))


class CellIdentifierManager:
    """
    A class to manage reading and writing CellIdentifier objects to a file.

    Attributes:
        file_path (str): The path to the file where the CellIdentifier objects will be stored.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def write_cells(self, cells: List[CellIdentifier]):
        """
        Write a list of CellIdentifier objects to the file.

        Args:
            cells (list): A list of CellIdentifier objects.
        """
        with open(self.file_path, 'w') as file:
            json.dump([cell.to_dict() for cell in cells], file)

        print('Written cell list to file!')

        
    def read_cells(self) -> List[CellIdentifier]:
        """
        Read CellIdentifier objects from the file.

        Returns:
            list: A list of CellIdentifier objects.
        """
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                return [CellIdentifier.from_dict(cell) for cell in data]
        except FileNotFoundError:
            return []

    def append_cells(self, new_cells: List[CellIdentifier]):
        """
        Append new CellIdentifier objects to the file.

        Args:
            new_cells (list): A list of new CellIdentifier objects to append.
        """
        cells = self.read_cells()

        overlap_cells = list(set(new_cells) & set(cells))
        overlap_cells_old = list(set(cells) & set(new_cells))

        if len(overlap_cells) != 0:
            print('Warning: detected cells that already exists in file! overwriting')
            print([f'{str(cell)}: Old label was {str(cell.label)}, new label is {str(old_cell.label)}' for cell, old_cell in zip(overlap_cells, overlap_cells_old)])

        cells = list(set(new_cells + cells))
        self.write_cells(cells)

    def generate_label_dict(self) -> dict:
        """
        Generate a dictionary with the specified label dict format.

        Returns:
            dict: The dictionary in the specified format.
        """
        cell_list = self.read_cells()

        label_dict = {}
        for cell in cell_list:
            fov_ext = f"{cell.proj_name}_{cell.fov}"
            if fov_ext not in label_dict:
                label_dict[fov_ext] = {}

            if cell.channel not in label_dict[fov_ext]:
                label_dict[fov_ext][cell.channel] = {}

            label_dict[fov_ext][cell.channel][cell.cell_id] = {
                'label': cell.label
            }
        
        return label_dict

    def get_fov_name_list(self) -> List[str]:
        cell_list = self.read_cells()
        fov_name_list = [cell.fov for cell in cell_list]
        return list(set(fov_name_list))
    

def get_path_to_label_csv(proj_name:str, get_with_filters:bool=False) -> str:
    """Generate path the marker positivity label csv file from project name.

    Args:
        proj_name (str): The name of the project.
        get_with_filters (bool, optional): if true, use csv file that includes cell filters to reduce label errors. Defaults to False.

    Returns:
        str: The path to the csv marker positivity label file
    """
    if get_with_filters:
        return os.path.join(get_wexac_base_path(), 'eliovits','marker_positivity_cnn','datasets','updated_marker_positivity_labels', proj_name+'_MarkerPositiveTrainingTableWithFilters.csv')
    else:
        return os.path.join(get_wexac_base_path(),  'Collaboration', 'CellTune_MarkerPositivityCNN', proj_name + '_markerPositiveTrainingTable.csv')


def get_parquet_cell_tbl_path(proj_name:str):
    return os.path.join(get_wexac_base_path(), 'eliovits', 'marker_positivity_cnn', 'projects_data', proj_name, 'cell_table_filtered.parquet')


def get_wexac_base_path() -> str:
    if platform == "linux" or platform == "linux2":
        base_wexac_path = '/home/labs/leeat'
    elif platform == "win32":
        base_wexac_path = r'Y:\\'
    return base_wexac_path


def get_df_cell_tbl(proj_name:str):
    path_to_cell_table_parquet = get_parquet_cell_tbl_path(proj_name=proj_name)
    return pd.read_parquet(path_to_cell_table_parquet)


def get_df_marker_positivity_labels(proj_name:str):
    path_to_label_csv = get_path_to_label_csv(proj_name=proj_name)
    return pl.read_csv(path_to_label_csv).to_pandas()


def get_merged_df_cell_tbl_and_marker_labels(proj_name: str):
    
    # Get the DataFrames
    df_cell_tbl = get_df_cell_tbl(proj_name=proj_name)
    df_labels = get_df_marker_positivity_labels(proj_name=proj_name)

    # Get relevant markers
    relevant_markers = df_labels['Marker'].unique()
    
    # Filter columns based on relevant markers
    filtered_columns = [col for col in df_cell_tbl.columns if col.split('_', 1)[0] in relevant_markers or col in ['fov', 'cellID']]
    df = df_cell_tbl[filtered_columns]

    id_columns = ['fov', 'cellID']
    columns = df.columns
    channel_names = sorted(list(set(col.split('_')[0] for col in columns if '_' in col)))
    property_names = sorted(list(set('_'.join(col.split('_')[1:]) for col in columns if '_' in col)))

    print("Channel Names:", channel_names)
    print("Property Names:", property_names)

    dfs = []

    # Iterate over each channel
    for channel in channel_names:
        # Filter columns that match the current channel
        channel_cols = [col for col in columns if col.startswith(channel+'_')]
        
        # Filter out the properties for this channel
        channel_properties = [col.split('_', 1)[1] for col in channel_cols]
        
        # Subset the DataFrame to include only the current channel's columns
        channel_df = df[id_columns + channel_cols].copy()
        
        # Rename columns to remove channel prefix
        new_columns = id_columns + channel_properties
        channel_df.columns = new_columns
        
        # Add the channel name as a new column
        channel_df['Marker'] = channel
        
        # Append the prepared DataFrame to the list
        dfs.append(channel_df)

    # Concatenate all the individual DataFrames into the final DataFrame
    final_df = pd.concat(dfs, ignore_index=True)

    # Sort columns for consistency
    final_df = final_df[['fov', 'cellID', 'Marker'] + property_names]

    # Merge with labels
    merged_df = pd.merge(final_df, df_labels, on=['fov', 'cellID', 'Marker'])

    print(f'Shape of melted cell table is {final_df.shape}')
    print(f'Shape of label table is {df_labels.shape}')
    print(f'Shape of merged table is {merged_df.shape}')
    print(f'Columns of merged table are {merged_df.columns}')

    return merged_df



def _fov_dir_has_segmentation_labels(fov_dir: str) -> bool:
    """True if ``fov_dir`` looks like a per-FOV image folder (seg label next to channels)."""
    for name in ("segmentation_labels.tiff", "segmentation_labels.tif"):
        if os.path.isfile(os.path.join(fov_dir, name)):
            return True
    return False


def list_fov_dirs_with_segmentation(images_root: str) -> List[str]:
    """
    Return sorted absolute paths to per-FOV directories under ``images_root``.

    - **Flat:** ``<images_root>/<FOV>/segmentation_labels.tiff`` (or ``.tif``).
    - **Nested:** if an immediate subdirectory has no segmentation file, subfolders one level
      deeper are treated as FOVs (e.g. ``<images_root>/<project>/<FOV>/``).

    Raises:
        FileNotFoundError: if ``images_root`` is missing.
        ValueError: if no FOV directory with ``segmentation_labels.tiff`` / ``.tif`` is found.
    """
    root = os.path.abspath(os.path.normpath(images_root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Not a directory: {root!r}")

    fovs: List[str] = []
    try:
        level1 = sorted(
            d
            for d in os.listdir(root)
            if not d.startswith(".") and os.path.isdir(os.path.join(root, d))
        )
    except OSError as e:
        raise FileNotFoundError(f"Cannot list directory: {root!r}") from e

    for name in level1:
        p = os.path.join(root, name)
        if _fov_dir_has_segmentation_labels(p):
            fovs.append(p)
            continue
        # Intermediate project folder — FOVs are usually one level down
        try:
            level2 = sorted(
                d
                for d in os.listdir(p)
                if not d.startswith(".") and os.path.isdir(os.path.join(p, d))
            )
        except OSError:
            continue
        for name2 in level2:
            q = os.path.join(p, name2)
            if _fov_dir_has_segmentation_labels(q):
                fovs.append(q)

    if not fovs:
        raise ValueError(
            f"No per-FOV folders with segmentation_labels.tiff under {root!r}. "
            "Expected either …/<FOV>/segmentation_labels.tiff or "
            "…/<intermediate>/<FOV>/segmentation_labels.tiff "
            "(see sample_data layout in the README)."
        )
    return sorted(fovs)


def get_segmentation_image(fov_image_dir_path, get_borders_image=False, seg_image_in_same_dir=True):
    if not get_borders_image:
        file_to_get = 'segmentation_labels.tiff'
    else:
        file_to_get = 'segmentation_borders.tiff'

    if seg_image_in_same_dir:
        seg_label_image_path = os.path.join(fov_image_dir_path, file_to_get)
    else:
        fov_name = os.path.basename(fov_image_dir_path)
        seg_label_image_path = os.path.join(os.path.dirname(fov_image_dir_path).replace('image_data', 'segmentation_data'),  fov_name+'_'+ file_to_get)

    # seg_label_image = Image.open(seg_label_image_path)
    # seg_label_image = np.array(seg_label_image.convert("I"))

    if not os.path.exists(seg_label_image_path):
        raise FileNotFoundError ('No segmentation image found at {}. seg_image_in_same_dir? {}'.format(seg_label_image_path, seg_image_in_same_dir))
    seg_label_image = cv2.imread(seg_label_image_path, cv2.IMREAD_UNCHANGED)
    
    # print(seg_label_image_path)
    return seg_label_image.astype(int)


def get_stacked_image_from_tiff(fov_image_dir_path, channel_list:List[str], seg_image_in_same_dir):
    seg_label_image = get_segmentation_image(fov_image_dir_path, seg_image_in_same_dir=seg_image_in_same_dir)

    stacked_img = np.zeros((seg_label_image.shape[0], seg_label_image.shape[1], len(channel_list)), dtype=np.uint8)
    for i, ch_name in enumerate(channel_list):
        try:
            image_path = glob.glob(os.path.join(fov_image_dir_path, ch_name + '.tif*'))[0]
        except IndexError as e:
            print(f'Warning: Could not find image for fov {os.path.basename(fov_image_dir_path)} channel {ch_name}')
            continue

        # image = Image.open(image_path)
        # image = image.convert("I")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        stacked_img[:, :, i] = image

    return stacked_img

def get_cell_props_from_fov(fov_image_dir_path):
    seg_label_image = get_segmentation_image(fov_image_dir_path)

    props = regionprops_table(seg_label_image,
                            properties=['label', 'area', 'bbox','orientation', 
                                         'axis_minor_length', 'axis_major_length'
                                        ])

    return props



def get_cropped_cell_from_props(img, cell_props, cell_idx, crop_size=128, augment_angle = 0, is_seg_img:bool=False):
    # ``np.where`` returns length-1 arrays; advanced indexing then yields 1-d bbox slices and
    # ``int(np.floor(...))`` raises. Scalar index is always required here.
    _ci = np.asarray(cell_idx).reshape(-1)
    if _ci.size != 1:
        raise ValueError(f"cell_idx must select exactly one cell; got {cell_idx!r}")
    cell_idx = int(_ci[0])

    h, w = img.shape
    half_width = int(crop_size//2)
    cell_bbox_row_min = np.floor(cell_props['bbox-0'][cell_idx])
    cell_bbox_row_max = np.floor(cell_props['bbox-2'][cell_idx])
    cell_bbox_col_min = np.floor(cell_props['bbox-1'][cell_idx])
    cell_bbox_col_max = np.floor(cell_props['bbox-3'][cell_idx])

    cell_center_row = int((cell_bbox_row_min + cell_bbox_row_max)//2)
    cell_center_col = int((cell_bbox_col_min + cell_bbox_col_max)//2)

    if augment_angle !=0:
        M = cv2.getRotationMatrix2D((cell_center_col,cell_center_row), augment_angle, 1.0)

        if is_seg_img:
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)
        else:
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        homogeneous_coordinate = np.array([cell_center_col, cell_center_row, 1])
        new_coordinate = np.dot(M, homogeneous_coordinate)
        cell_center_col, cell_center_row = new_coordinate[:2]
        cell_center_row = int(cell_center_row)
        cell_center_col = int(cell_center_col)

    final_crop = np.zeros((crop_size, crop_size))

    ind_start_crop_row = cell_center_row-half_width
    if ind_start_crop_row < 0:
        ind_start_crop_row = 0
    ind_start_assign_row = ind_start_crop_row - (cell_center_row-half_width)

    ind_end_crop_row = cell_center_row+half_width
    if ind_end_crop_row > h:
        ind_end_crop_row = h
    ind_end_assign_row = crop_size- ((cell_center_row+half_width) - ind_end_crop_row)

    ind_start_crop_col = cell_center_col-half_width
    if ind_start_crop_col < 0:
        ind_start_crop_col = 0
    ind_start_assign_col = ind_start_crop_col - (cell_center_col-half_width)

    ind_end_crop_col = cell_center_col+half_width
    if ind_end_crop_col > w:
        ind_end_crop_col = w
    ind_end_assign_col = crop_size - ((cell_center_col+half_width) - ind_end_crop_col)


    # Crop Cell
    final_crop[ind_start_assign_row:ind_end_assign_row, ind_start_assign_col: ind_end_assign_col] = img[ind_start_crop_row:ind_end_crop_row, ind_start_crop_col:ind_end_crop_col]

    return final_crop


def get_cell_crops(fov_img, cell_props, half_width=64):
    tic = time.time()
    n_channels = fov_img.shape[2]
    n_cells = len(cell_props['label'])
    bbox_min_row = np.floor(cell_props['bbox-0']).astype(int)
    bbox_min_col = np.floor(cell_props['bbox-1']).astype(int)
    bbox_h = np.floor(cell_props['bbox-2'] - cell_props['bbox-0']).astype(int)
    bbox_w = np.floor(cell_props['bbox-3'] - cell_props['bbox-1']).astype(int)
    orientation = cell_props['orientation'] * 180 / np.pi
    min_len = cell_props['axis_minor_length']
    maj_len = cell_props['axis_major_length']

    buffer_px = int(5*(np.ceil(max(np.max(bbox_h), np.max(bbox_w))/10)+1)*10)

    bbox_min_row += buffer_px
    bbox_min_col += buffer_px


    r_idx_start = (bbox_min_row - bbox_h*3)-1
    r_idx_stop = (bbox_min_row + bbox_h*4)
    c_idx_start = (bbox_min_col - bbox_w*3)-1
    c_idx_stop = (bbox_min_col + bbox_w*4)
    

    # Initial Crop
    cell_crop_areas = [fov_img[r_idx_start[i]:r_idx_stop[i], c_idx_start[i]:c_idx_stop[i]] for i in range(n_cells)]

    # Rotate
    rotated_cell_crops = [rotate(input=cell_crop_areas[i], angle=-orientation[i]+90, reshape=False, order=0, prefilter=False) for i in range(n_cells)]

    # Resize
    r_size = [int(np.ceil(rotated_cell_crops[i].shape[0] / (min_len[i] / 32))) for i in range(n_cells)]
    c_size = [int(np.ceil(rotated_cell_crops[i].shape[1] / (maj_len[i] / 32))) for i in range(n_cells)]

    # resized_cell_crops = [cv2.resize(rotated_cell_crops[i], (np.maximum(c_size[i], half_width*2), np.maximum(r_size[i], half_width*2)), interpolation=cv2.INTER_NEAREST) for i in range(n_cells)]


    resized_cell_crops = [skimage.transform.resize(rotated_cell_crops[i], (np.maximum(r_size[i], half_width*2), np.maximum(c_size[i], half_width*2)), order=0, mode='constant') for i in range(n_cells)]

    # Final crop
    r_cent = np.floor(np.maximum(r_size, half_width*2) / 2).astype(int)
    c_cent = np.floor(np.maximum(c_size, half_width*2) / 2).astype(int)

    cell_crops_final_crop = [resized_cell_crops[i][(r_cent[i]-half_width):(r_cent[i]+half_width), (c_cent[i]-half_width):(c_cent[i]+half_width)] for i in range(n_cells)]

    # Convert from float to int16
    cell_crops_final_crop = [(cell_crops_final_crop[i] * 256).astype(np.int16) for i in range(n_cells)]
    cell_crops_final_crop = np.array(cell_crops_final_crop) 
    toc = time.time()
    print(f'time to get all cell crops for fov is {toc - tic}')
    return cell_crops_final_crop


def pad_fov_image(fov_img, cell_props):
    n_cells = len(cell_props['label'])
    bbox_min_row = np.floor(cell_props['bbox-0']).astype(int)
    bbox_min_col = np.floor(cell_props['bbox-1']).astype(int)
    bbox_h = np.floor(cell_props['bbox-2'] - cell_props['bbox-0']).astype(int)
    bbox_w = np.floor(cell_props['bbox-3'] - cell_props['bbox-1']).astype(int)
    orientation = cell_props['orientation'] * 180 / np.pi
    min_len = cell_props['axis_minor_length']
    maj_len = cell_props['axis_major_length']

    buffer_px = int(5*(np.ceil(max(np.max(bbox_h), np.max(bbox_w))/10)+1)*10)

    bbox_min_row += buffer_px
    bbox_min_col += buffer_px

    if len(fov_img.shape) == 3:

        padded_fov_img = np.pad(fov_img, pad_width=((buffer_px, buffer_px), (buffer_px, buffer_px), (0, 0)))
    else:
        padded_fov_img = np.pad(fov_img, pad_width=((buffer_px, buffer_px), (buffer_px, buffer_px)))

    return padded_fov_img

def get_single_cell_crop(fov_img, cell_props, cell_idx, half_width=64):
    if type(cell_idx) == np.ndarray:
        cell_idx = cell_idx[0]
    n_cells = len(cell_props['label'])
    bbox_min_row = np.floor(cell_props['bbox-0']).astype(int)
    bbox_min_col = np.floor(cell_props['bbox-1']).astype(int)
    bbox_h = np.floor(cell_props['bbox-2'] - cell_props['bbox-0']).astype(int)
    bbox_w = np.floor(cell_props['bbox-3'] - cell_props['bbox-1']).astype(int)
    orientation = cell_props['orientation'] * 180 / np.pi
    min_len = cell_props['axis_minor_length']
    maj_len = cell_props['axis_major_length']

    if min_len[cell_idx] == 0 or maj_len[cell_idx] == 0 or bbox_h[cell_idx] == 0 or bbox_w[cell_idx] == 0:
        return np.zeros((half_width*2, half_width*2)).astype(np.int16)

    buffer_px = int(5*(np.ceil(max(np.max(bbox_h), np.max(bbox_w))/10)+1)*10)

    bbox_min_row += buffer_px
    bbox_min_col += buffer_px

    r_idx_start = (bbox_min_row - bbox_h*3)-1
    r_idx_stop = (bbox_min_row + bbox_h*4)
    c_idx_start = (bbox_min_col - bbox_w*3)-1
    c_idx_stop = (bbox_min_col + bbox_w*4)

    # Initial Crop
    cell_crop_area = fov_img[r_idx_start[cell_idx]:r_idx_stop[cell_idx], c_idx_start[cell_idx]:c_idx_stop[cell_idx]]

    # Rotate
    rotated_cell_crop = rotate(input=cell_crop_area, angle=-orientation[cell_idx]+90, reshape=False, order=0, prefilter=False)

    # Resize
    r_size = int(np.ceil(rotated_cell_crop.shape[0] / (min_len[cell_idx] / 32)))
    c_size = int(np.ceil(rotated_cell_crop.shape[1] / (maj_len[cell_idx] / 32)))

    resized_cell_crop = cv2.resize(rotated_cell_crop, (np.maximum(c_size, half_width*2), np.maximum(r_size, half_width*2)), interpolation=cv2.INTER_NEAREST)

    # Final crop
    r_cent = np.floor(np.maximum(r_size, half_width*2) / 2).astype(int)
    c_cent = np.floor(np.maximum(c_size, half_width*2) / 2).astype(int)

    cell_crops_final_crop = resized_cell_crop[(r_cent-half_width):(r_cent+half_width), (c_cent-half_width):(c_cent+half_width)]

    # Convert from float to int16
    cell_crops_final_crop = (cell_crops_final_crop * 256).astype(np.int16)
    cell_crops_final_crop = np.array(cell_crops_final_crop) 
    cell_crops_final_crop_PIL = Image.fromarray(cell_crops_final_crop)
    return cell_crops_final_crop_PIL


if __name__ == "__main__":
    get_merged_df_cell_tbl_and_marker_labels(proj_name='PDAC_Aug1423')