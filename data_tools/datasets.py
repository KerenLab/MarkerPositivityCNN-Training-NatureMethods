from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os
import random
import time
from typing import Dict, List, Union

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from skimage.measure import regionprops_table
from tifffile import TiffFile
from torch.utils.data import Dataset
from torchvision.transforms import v2

from data_tools.channels_data import MarkerType, get_marker_type_encoding
from data_tools.dataset_utils import (get_cropped_cell_from_props,
                                      get_segmentation_image,
                                      get_single_cell_crop,
                                      get_stacked_image_from_tiff,
                                      pad_fov_image)


def create_train_indices_random(x=0.75, seed=42):
    df = pd.read_csv('mpcnn_patient_info_Mar0625.csv')
    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    dataset_indices = {}
    for dataset in df['dataset'].unique():
        # Filter and sort the dataset
        df_dataset = df[df['dataset'] == dataset].copy()
        df_sorted = df_dataset.sort_values('fov').reset_index(drop=True)
        
        # Get unique patient IDs and shuffle them randomly
        unique_patients = df_sorted['patientID'].unique()
        random.shuffle(unique_patients)
        
        # Calculate target number of FOVs
        total_fovs = len(df_sorted)
        target_fovs = x * total_fovs
        
        # Select patients for training
        cumulative = 0
        train_patients = []
        for patient in unique_patients:
            # Number of FOVs for this patient
            count = len(df_sorted[df_sorted['patientID'] == patient])
            # Check if adding this patient would exceed the target (if not the first patient)
            if cumulative + count > target_fovs and cumulative != 0:
                break
            train_patients.append(patient)
            cumulative += count
            # Stop if target is met or exceeded
            if cumulative >= target_fovs:
                break
        
        # Extract indices of training FOVs
        train_indices = df_sorted.index[df_sorted['patientID'].isin(train_patients)].to_numpy()
        dataset_indices[dataset] = train_indices
        val_patients = sorted(list(set(unique_patients) - set(train_patients)))
        print(f'For {dataset}, choose patients {sorted(train_patients)}, val Patients are {val_patients}, representing {len(train_indices)/len(df_sorted)*100}% of Fovs')

    return dataset_indices

def create_train_indices(x=0.75):
    df = pd.read_csv('mpcnn_patient_info_Mar0625.csv')
    dataset_indices = {}
    for dataset in df['dataset'].unique():
        # Filter and sort the dataset
        df_dataset = df[df['dataset'] == dataset].copy()
        df_sorted = df_dataset.sort_values('fov').reset_index(drop=True)
        
        # Determine the order of unique patientIDs as they first appear in the sorted data
        unique_patients = []
        for _, row in df_sorted.iterrows():
            patient_id = row['patientID']
            if patient_id not in unique_patients:
                unique_patients.append(patient_id)
        
        # Calculate target number of FOVs
        total_fovs = len(df_sorted)
        target_fovs = x * total_fovs
        
        # Select patients for training
        cumulative = 0
        train_patients = []
        for patient in unique_patients:
            # Number of FOVs for this patient
            count = len(df_sorted[df_sorted['patientID'] == patient])
            # Check if adding this patient would exceed the target (if not the first patient)
            if cumulative + count > target_fovs and cumulative != 0:
                break
            train_patients.append(patient)
            cumulative += count
            # Stop if target is met or exceeded
            if cumulative >= target_fovs:
                break
        
        # Extract indices of training FOVs
        train_indices = df_sorted.index[df_sorted['patientID'].isin(train_patients)].to_numpy()
        dataset_indices[dataset] = train_indices
    
    return dataset_indices

def check_patient_overlap():
    df = pd.read_csv('mpcnn_patient_info_Mar0625.csv')
    for dataset in df['dataset'].unique():
        # Filter and sort the dataset
        df_dataset = df[df['dataset'] == dataset].copy()
        df_sorted = df_dataset.sort_values('fov').reset_index(drop=True)
        total_fovs = len(df_sorted)
        split_idx = int(0.75 * total_fovs)
        
        # Split into train and test
        train_fovs = df_sorted.iloc[:split_idx]
        test_fovs = df_sorted.iloc[split_idx:]
        
        # Find unique patients in each set
        train_patients = set(train_fovs['patientID'].unique())
        test_patients = set(test_fovs['patientID'].unique())
        shared_patients = train_patients.intersection(test_patients)
        
        # Print the results
        print(f"For dataset {dataset}, the training set contains fovs from {len(train_patients)} unique patients, "
              f"and the test set contains {len(test_patients)} unique patients, "
              f"out of which {len(shared_patients)} are shared and appear in both.")
        


def get_dataset_weights(
    label_dict, print_dataset_stats: bool = False, *, verbose: bool = True
):
    total_negative_and_not_trained_on = 0
    total_negative_and_trained_on = 0
    total_positive_and_not_trained_on = 0
    total_positive_and_trained_on = 0
    for fov_name in label_dict.keys():
        for channel_name in label_dict[fov_name].keys():
            for cell_label in label_dict[fov_name][channel_name].keys():
                if label_dict[fov_name][channel_name][cell_label]['was_trained_on']:
                    if label_dict[fov_name][channel_name][cell_label]['label'] == 0:
                        total_negative_and_trained_on += 1
                    else:
                        total_positive_and_trained_on += 1
                else:
                    if label_dict[fov_name][channel_name][cell_label]['label'] == 0:
                        total_negative_and_not_trained_on += 1
                    else:
                        total_positive_and_not_trained_on += 1

    if verbose:
        print(f'Total positive: {total_positive_and_trained_on + total_positive_and_not_trained_on}')
        print(f'Total negative: {total_negative_and_trained_on + total_negative_and_not_trained_on}')
    p_pos = 0.45
    p_neg = 1-p_pos
    p_t = 0.98
    p_nt = (1-p_t)

    pp_t = 0.8
    pp_nt = 1-pp_t

    def protected_inverse(x):
        if type(x) == np.ndarray:
            xf = np.asarray(x, dtype=float)
            # Avoid RuntimeWarning: divide by zero (zeros → 0, not inf)
            return np.where(xf != 0, 1.0 / xf, 0.0)

        if x != 0:
            return 1 / x
        if verbose:
            print('Warining: zero detected in denominator while calculating class weights for dataloader')
        return 0

    w_p_t = p_pos * pp_t * protected_inverse(total_positive_and_trained_on)
    w_p_nt = p_pos * pp_nt * protected_inverse(total_positive_and_not_trained_on)
    w_n_t = p_neg * p_t * protected_inverse(total_negative_and_trained_on)
    w_n_nt = p_neg * p_nt * protected_inverse(total_negative_and_not_trained_on)

    weight_list = list()

    for fov_name in label_dict.keys():
        for channel_name in label_dict[fov_name].keys():
            for cell_label in label_dict[fov_name][channel_name].keys():
                if label_dict[fov_name][channel_name][cell_label]['was_trained_on']:
                    if label_dict[fov_name][channel_name][cell_label]['label'] == 0:
                        weight_list.append(w_n_t)
                    else:
                        weight_list.append(w_p_t)
                else:
                    if label_dict[fov_name][channel_name][cell_label]['label'] == 0:
                        weight_list.append(w_n_nt)
                    else:
                        weight_list.append(w_p_nt)

    tot_w = np.sum(np.array(weight_list))
    p_p_t = w_p_t * protected_inverse(tot_w)*total_positive_and_trained_on
    p_p_nt = w_p_nt * protected_inverse(tot_w)*total_positive_and_not_trained_on
    p_n_t = w_n_t * protected_inverse(tot_w)*total_negative_and_trained_on
    p_n_nt = w_n_nt * protected_inverse(tot_w)*total_negative_and_not_trained_on

    class_sizes = np.array([total_positive_and_trained_on, total_positive_and_not_trained_on, total_negative_and_trained_on, total_negative_and_not_trained_on])
    class_tot_probs = np.array([p_p_t, p_p_nt, p_n_t, p_n_nt])
    smallest_class_ind = np.argmin(class_sizes)
    n_samples = int((class_sizes * protected_inverse(class_tot_probs))[smallest_class_ind])
    # Avoid 0/NaN when a class count is 0 or probabilities collapse (else samplers become num_samples=1).
    if not np.isfinite(n_samples) or n_samples < 1:
        n_samples = max(1, int(np.sum(class_sizes) // 10))

    if print_dataset_stats:
        print(f'Total weight sum: {tot_w}')
        print(f'Number of positive samples in the dataset: {total_positive_and_trained_on}')
        print(f'weight assigned to positive samples: {w_p_t}')
        print(f'prob to get positive samples: {p_p_t}')

        print(f'Number of new positive samples in the dataset: {total_positive_and_not_trained_on}')
        print(f'weight assigned to new positive samples: {w_p_nt}')
        print(f'prob to get new positive samples: {p_p_nt}')

        print(f'Number of negative samples in the dataset: {total_negative_and_trained_on}')
        print(f'weight assigned to negative samples: {w_n_t}')
        print(f'prob to get negative samples: {p_n_t}')

        print(f'Number of new negative samples in the dataset: {total_negative_and_not_trained_on}')
        print(f'weight assigned to new negative samples: {w_n_nt}')
        print(f'prob to get new negative samples: {p_n_nt}')

        print(f'suggested number of samples for this dataset is {n_samples}')

    return  weight_list, n_samples


def parse_label_csv_files(label_csv_path_list: List[str],
                           fov_dir_to_filter: List[np.ndarray],
                             channel_list_to_filter: Union[List[str], List[MarkerType]]=None,
                              keep_only_manual: bool = False,
                                get_only_baseline_validation_samples:bool=False,
                                verbose: bool = True) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:

    if verbose:
        print('Parsing label csv files...')
    label_dict = {}
    all_fov_names = list()
    channel_list_all = list()
    total_manual = 0
    total_was_trained_on = 0
    total_positive = 0
    total_negative = 0
    total_negative_and_trained_on = 0
    total_negative_and_not_trained_on = 0
    if fov_dir_to_filter is not None:
        fov_names_to_keep =[os.path.basename(fov_dir_to_filter[i]) for i in range(len(fov_dir_to_filter))]
    for i, label_csv_path in enumerate(label_csv_path_list):

        proj_name = os.path.basename(label_csv_path).split('_')
        proj_name = proj_name[0] + '_' + proj_name[1]
        df = pd.read_csv(label_csv_path)
        all_fov_names += list(df['fov'].unique())

        if fov_dir_to_filter is not None:
            if verbose:
                print('Filtering label csv by FOVs:')
                print('len label csv before filtering fovs: ', len(df))
            df = df[df['fov'].isin(fov_names_to_keep)]
            if verbose:
                print('len label csv after filtering fovs: ', len(df))
        if channel_list_to_filter is not None:
            if type(channel_list_to_filter[0]) == MarkerType:
                df = df[get_marker_type_encoding(df['Marker']).isin(channel_list_to_filter)]
            else:
                if verbose:
                    print('Filtering label csv by Markers')
                    print('Removing the markers ', channel_list_to_filter)
                    print('len label csv before filtering markers: ', len(df))
                df = df[~df['Marker'].isin(channel_list_to_filter)]
                if verbose:
                    print('len label csv after filtering markers: ', len(df))

        if keep_only_manual:
            df = df[df['Manual'] == 1]

        if get_only_baseline_validation_samples:
            df = df[df['InTraining'] == 0]

        fovs = df['fov']
        cell_ids = df['cellID']
        channel_names = df['Marker']
        labels = df['Positive']
        manual_labels = df['Manual']
        was_trained_on = df['InTraining']

        total_manual += len(df[manual_labels == 1])
        total_was_trained_on += len(df[was_trained_on == 1])
        total_positive += len(df[(labels==1) & (manual_labels == 0)])
        total_negative += len(df[(labels==-1) & (manual_labels == 0)])
        
        total_negative_and_trained_on += len(df[(labels == -1) & (was_trained_on == 1) & (manual_labels == 0)])
        total_negative_and_not_trained_on += len(df[(labels == -1) & (was_trained_on == 0) & (manual_labels == 0)])

        channel_list = channel_names.unique()
        channel_list_all += list(channel_list)
        
        for fov, cell_id, channel_name, label, manual_label, trained in tqdm.tqdm(
            zip(fovs, cell_ids, channel_names, labels, manual_labels, was_trained_on),
            desc='csv rows',
            total=len(fovs),
            disable=not verbose,
        ):
            fov_ext = proj_name + '_' + fov
            if fov_ext not in label_dict:
                label_dict[fov_ext] = {}

            if channel_name not in label_dict[fov_ext]:
                label_dict[fov_ext][channel_name] = {}


            label_dict[fov_ext][channel_name][cell_id] = {}
            label_dict[fov_ext][channel_name][cell_id]['label'] = (1 if (label == 1) else 0 )
            label_dict[fov_ext][channel_name][cell_id]['Manual_label'] = manual_label
            label_dict[fov_ext][channel_name][cell_id]['was_trained_on'] = trained

    weight_list, n_samples = get_dataset_weights(label_dict, verbose=verbose)
    if verbose:
        print(f"All channels detected {len(set(channel_list_all))}")
    return label_dict, np.array(weight_list), n_samples


class FOVDataset(Dataset):
    """A general class for cell intensity image dataset, based on full field of view image directories. The class handles the creation of proper cell crops from multiple channels, dataset enumeration and labels. 

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, manits_img_dir_path_list, seg_as_channels:bool=True, get_metadata:bool=True, add_augmentations:bool=False, cache_images:bool=False, channel_list=[], only_labels:bool=False, seg_image_is_in_same_dir_as_other_images:bool = True, verbose: bool = True):
        self.manits_img_dir_path_list = manits_img_dir_path_list
        self.get_metadata = get_metadata
        self.seg_as_channels = seg_as_channels
        self.b_cache_images=cache_images
        self.channel_list=channel_list
        self.seg_image_in_same_dir = seg_image_is_in_same_dir_as_other_images
        self.verbose = verbose
        self.seg_images_cache = list()
        if not only_labels:
            self.calc_cell_props()
        self.add_augmentations = add_augmentations
        self.cached_images = {}
        self.max_cache_size = 2
        if not seg_as_channels:
            # Use single channle intensity images as input - cells will be transformed into a standard shape.
            if add_augmentations:
                print('Warning: augmentations not currently supported for the non segmentation dataset type')

            self.out_transform = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), 
                                    ScaleImageuint16()
                                    ])
        else:
            # Use single channle intensity images as input - cells will be transformed into a standard shape. 
            if add_augmentations:
                # Note that rotation augmentation does exist but is implemented separately 
                self.out_transform = transforms.Compose([
                                                ParticleResampling(probability=0.5),
                                                transforms.ToTensor(),
                                                v2.RandomHorizontalFlip(p=0.5),
                                                ])
            else:
                self.out_transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])


    def get_fov_image_from_cache(self, fov_img_ind, channel, pad_image=True):
        if type(channel) is str:
            channel_ind = self.channel_list.index(channel)
        else:
            channel_ind = channel
        key = str(fov_img_ind) + '_' + str(channel_ind)
        if key not in self.cached_images.keys():
            fov_image = get_stacked_image_from_tiff(self.manits_img_dir_path_list[fov_img_ind], [channel], self.seg_image_in_same_dir)
            if pad_image:
                fov_image = pad_fov_image(fov_image, self.cell_props_list[fov_img_ind]) 

            if len(self.cached_images.keys()) >= self.max_cache_size:
                del self.cached_images[list(self.cached_images.keys())[0]]

            self.cached_images[key] = fov_image[:, :, 0]
        return self.cached_images[key]


    def calc_cell_props(self):
        self.cell_props_by_fov = {}
        if self.seg_as_channels:
            properties =  ['label','bbox']
        else:
            properties=['label', 'area', 'bbox','orientation', 
                'axis_minor_length', 'axis_major_length'
            ]
        if self.verbose:
            print('Loading segmentation images. ')
        self.seg_images_cache = [None] * len(self.manits_img_dir_path_list)  # Preallocate list to maintain order

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(get_segmentation_image, path, seg_image_in_same_dir=self.seg_image_in_same_dir): idx for idx, path in enumerate(self.manits_img_dir_path_list)}
        
        for future in as_completed(futures):
            idx = futures[future]  # Retrieve index of the completed future
            self.seg_images_cache[idx] = future.result()  # Store the result in the correct position

        if self.verbose:
            print('Calculating cell props...')
        self.cum_cell_counts = list()

        self.cell_props_list = [None] * len(self.manits_img_dir_path_list)  # Preallocate list to maintain order

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(regionprops_table, self.seg_images_cache[idx], properties= properties, cache=True): idx for idx, path in enumerate(self.manits_img_dir_path_list)}
        
        for future in as_completed(futures):
            idx = futures[future]  # Retrieve index of the completed future
            self.cell_props_list[idx] = future.result()  # Store the result in the correct position



        for i, manits_img_dir_path in tqdm.tqdm(
            enumerate(self.manits_img_dir_path_list),
            desc='FOVs in dataset',
            total=len(self.manits_img_dir_path_list),
            disable=not self.verbose,
        ):
            self.cell_props_by_fov[i] = self.cell_props_list[i]
            self.cum_cell_counts.append(len(self.cell_props_by_fov[i]['label'])*len(self.channel_list))
        self.cum_cell_counts = np.array(self.cum_cell_counts)
        self.cum_cell_counts = np.cumsum(self.cum_cell_counts)


    def get_cropped_cell_image_with_seg(self, fov_name, channel_name, cell_label, crop_size=128):
        # Image rotation augmentation
        if self.add_augmentations &  (np.random.rand() > 0.5):
            angle = random.uniform(-180, 180)
        else:
            angle = 0

        fov_img_ind = self.get_fov_ind_from_fov_name(fov_name)
        cell_idx_in_channel = np.where(self.cell_props_by_fov[fov_img_ind]['label'] == cell_label)[0]

        # Get intensity image
        if self.b_cache_images:
            im = self.get_fov_image_from_cache(fov_img_ind, channel=channel_name, pad_image=False)
            cropped_cell = get_cropped_cell_from_props(im, self.cell_props_by_fov[fov_img_ind], cell_idx_in_channel, crop_size, augment_angle=angle)
        else:
            image_path = glob.glob(os.path.join(self.manits_img_dir_path_list[fov_img_ind], channel_name + '.tif*'))[0]
            im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            cropped_cell = get_cropped_cell_from_props(im, self.cell_props_by_fov[fov_img_ind], cell_idx_in_channel, crop_size, augment_angle=angle)

        # Get segmentation image
        if self.b_cache_images:
            seg_image = self.seg_images_cache[fov_img_ind]
        else:
            seg_image = get_segmentation_image(self.manits_img_dir_path_list[fov_img_ind])

        cropped_seg = get_cropped_cell_from_props(seg_image, self.cell_props_by_fov[fov_img_ind], cell_idx_in_channel, crop_size, augment_angle=angle, is_seg_img=True)


        cell_label_id = self.cell_props_by_fov[fov_img_ind]['label'][cell_idx_in_channel]
        cropped_seg_cell = ((cropped_seg == cell_label_id) * 32).astype(np.uint8)
        cropped_seg_neighbors =(((cropped_seg != 0) &
                                 (cropped_seg != cell_label_id))* 32).astype(np.uint8)
        crop_3_channels = np.stack([cropped_cell.astype(np.uint8)* 32, cropped_seg_cell, cropped_seg_neighbors], axis=-1)

        return crop_3_channels # Note - output img shape = H, W, C (will later be transformed by out_transform to C,H,W)


    def get_cropped_cell_with_transform(self, fov_name, channel_name, cell_label):
        fov_img_ind = self.get_fov_ind_from_fov_name(fov_name)
        cell_idx_in_channel = np.where(self.cell_props_by_fov[fov_img_ind]['label'] == cell_label)[0]

        # Read image
        if self.b_cache_images:
            im = self.get_fov_image_from_cache(fov_img_ind, channel=channel_name, pad_image=True)
        else:
            image_path = glob.glob(os.path.join(self.manits_img_dir_path_list[fov_img_ind], channel_name + '.tif*'))[0]
            im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            im = pad_fov_image(im, self.cell_props_by_fov[fov_img_ind]) 
        im = im[..., np.newaxis]


        cropped_cell= get_single_cell_crop(im, self.cell_props_by_fov[fov_img_ind], cell_idx=cell_idx_in_channel)
        return cropped_cell

    def get_fov_ind_from_fov_name(self, fov_name):
        fov_names_ext = [fov_path.split(os.sep)[-3] + '_' + os.path.basename(fov_path) for fov_path in self.manits_img_dir_path_list]
        return fov_names_ext.index(fov_name)


    def get_cell_by_att(self, fov_name, channel_name, cell_id):
        if self.seg_as_channels:
            cell_crop = self.get_cropped_cell_image_with_seg(fov_name, channel_name, cell_id)
        else:
            cell_crop = self.get_cropped_cell_with_transform(fov_name, channel_name, cell_id)

        if self.labels_dict is not None:
            cell_label = self.labels_dict[fov_name][channel_name][cell_id]['label']
        else:
            cell_label = -1
        if self.get_metadata:
            cell_data = fov_name, channel_name, cell_id, get_marker_type_encoding(channel_name)

            return self.out_transform(cell_crop), cell_label, cell_data
        else:
            return self.out_transform(cell_crop)        

    def get_cell_label_by_att(self, fov_name, channel_name, cell_id):
        if self.labels_dict is not None:
            cell_label = self.labels_dict[fov_name][channel_name][cell_id]['label']
        else:
            cell_label = -1
        return cell_label

    def __len__(self):
        return

    def __getitem__(self, idx):
        return

class FOVDatasetByMantisFolder(FOVDataset):
    """ A class for cell intensity image dataset, based on full field of view image directories.
    The class handles the creation of proper cell crops from multiple channels, dataset enumeration and labels.
    This class uses all of the cells in a given fov directory ("mantis directory"), regardless of the existence of a marker positivity label. 
    Useful for general prediction purposes.
    """

    def __init__(self, manits_img_dir_path_list, channel_list, labels_dict, seg_as_channels:bool=True, get_metadata=True, add_augmentations:bool=False, cache_images:bool=True, seg_image_is_in_same_dir_as_other_images:bool = True):
        """
        Args:
            image_paths (list of str): List of file paths for the full FOV images.
            labels (list of int): List of labels for the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(FOVDatasetByMantisFolder, self).__init__(manits_img_dir_path_list, seg_as_channels, get_metadata, add_augmentations, cache_images, channel_list, seg_image_is_in_same_dir_as_other_images=seg_image_is_in_same_dir_as_other_images)

        self.channel_list = channel_list
        self.labels_dict=labels_dict

    def __len__(self):
        return self.cum_cell_counts[-1]
    


    def __getitem__(self, idx):
        fov_img_ind = np.where(np.array(self.cum_cell_counts) > idx)[0][0]

        cell_props = self.cell_props_list[fov_img_ind]
        num_cells_in_fov = len(cell_props['label'])


        if fov_img_ind != 0:
            idx_in_fov = idx - self.cum_cell_counts[fov_img_ind-1]
        else:
            idx_in_fov = idx
        channel_ind = idx_in_fov // num_cells_in_fov
        cell_idx_in_channel = idx_in_fov % num_cells_in_fov


        fov_path = self.manits_img_dir_path_list[fov_img_ind]
        fov_name_ext = fov_path.split(os.sep)[-3] + '_' + os.path.basename(fov_path)
        channel_name = self.channel_list[channel_ind]
        cell_id =cell_props['label'][cell_idx_in_channel]
        try:
            label = self.labels_dict[fov_name_ext][channel_name][cell_id]['label']
        except Exception as e:
            label=-1

        if self.seg_as_channels:
            cell_crop = self.get_cropped_cell_image_with_seg(fov_name_ext, channel_name, cell_id)
        else:
            fov_image = self.get_fov_image_from_cache(fov_img_ind, channel_name)
            cell_crop = get_single_cell_crop(fov_image, self.cell_props_list[fov_img_ind], cell_idx=cell_idx_in_channel)

        cell_crop = self.out_transform(cell_crop)


        if self.get_metadata:
            channel_name = self.channel_list[channel_ind]
            cell_id = cell_props['label'][cell_idx_in_channel]
            # fov_name = os.path.basename(self.manits_img_dir_path_list[fov_img_ind])
            cell_data = fov_name_ext, channel_name, cell_id, get_marker_type_encoding(channel_name)
            return cell_crop, label, cell_data#, str(fov_img_ind) + '_' + str(channel_ind)
        else:
            return cell_crop, label



class FOVDatasetByLabelDict(FOVDataset):
    """ A class for cell intensity image dataset, based on full field of view image directories.
    The class handles the creation of proper cell crops from multiple channels, dataset enumeration and labels.
    This class builds a dataset according to a "labels_dict", and will only contain cells with label. 
    Useful for model training and validation purposes.
    """

    def __init__(self,
                  manits_img_dir_path_list: List[str],
                    labels_dict: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
                    get_metadata:bool=True,
                    add_augmentations:bool=False,
                    seg_as_channels:bool=True,
                    cache_images:bool=False,
                    only_labels: bool=False,
                    verbose: bool = True):
        super(FOVDatasetByLabelDict, self).__init__(manits_img_dir_path_list=manits_img_dir_path_list,
                                                     seg_as_channels=seg_as_channels,
                                                       get_metadata=get_metadata,
                                                         add_augmentations=add_augmentations,
                                                           cache_images=cache_images,
                                                            only_labels=only_labels,
                                                            verbose=verbose)

        self.labels_dict=labels_dict
        self.order_cells()
        self.only_label = only_labels

    def order_cells(self):
        self.cell_origin_list = list()
        self.channel_list = list()
        if self.verbose:
            print('Sorting cells...')
        for fov_name in tqdm.tqdm(self.labels_dict.keys(), disable=not self.verbose):
            self.channel_list = list(set(self.channel_list + list(self.labels_dict[fov_name].keys())))
            for channel_name in self.labels_dict[fov_name].keys():
                for cell_label in self.labels_dict[fov_name][channel_name].keys():
                    self.cell_origin_list.append((fov_name, channel_name, cell_label))
                


    def __len__(self):
        return len(self.cell_origin_list)
    

    
    def __getitem__(self, idx):
        if self.only_label:
            return self.get_cell_label_by_cell_idx(idx)
        
        fov_name, channel_name, cell_id = self.cell_origin_list[idx]
        cell_label = self.labels_dict[fov_name][channel_name][cell_id]['label']
        cell_data = fov_name, channel_name, cell_id, int(get_marker_type_encoding(channel_name))

        if self.seg_as_channels:
            cell_crop = self.get_cropped_cell_image_with_seg(fov_name, channel_name, cell_id)
            return self.out_transform(cell_crop), cell_label, cell_data
        else:
            cell_crop = self.get_cropped_cell_with_transform(fov_name, channel_name, cell_id)
            try:
                return self.out_transform(cell_crop), cell_label, cell_data
            except TypeError as e:
                print(f'Cell crop type is {type(cell_crop)}, cell crop shape is {cell_crop.shape}')
                cell_crop = Image.fromarray(cell_crop)
                return self.out_transform(cell_crop), cell_label, cell_data


    def get_cell_label_by_cell_idx(self, idx):
        fov_name, channel_name, cell_id = self.cell_origin_list[idx]
        return super().get_cell_label_by_att(fov_name, channel_name, cell_id), fov_name, channel_name, cell_id


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ScaleImageuint16(object):
    def __init__(self):
        return
    def __call__(self, tensor):
        return tensor.type(torch.float32) / 65535.0
    
    def __repr__(self):
        return self.__class__.__name__ 

class ScaleImageuint8(object):
    def __init__(self, mean_norm:bool = False):
        self.mean_norm = mean_norm

    def __call__(self, tensor):
        tensor_normed = tensor.type(torch.float32)
        tensor_normed[0, ...] = tensor_normed[0, ...]/255 * 10
        tensor_normed[1, ...] = tensor_normed[1, ...]/255
        tensor_normed[2, ...] = tensor_normed[2, ...]/255

        if self.mean_norm:
            mean_val = torch.quantile(tensor_normed[0, ...], torch.tensor([0.5]))
            tensor_normed[1, ...] = tensor_normed[1, ...] * 25 * mean_val
            tensor_normed[2, ...] = tensor_normed[2, ...] * 25 * mean_val
            tensor_normed[0, ...] = tensor_normed[0, ...] - mean_val

        return tensor_normed
    
    def __repr__(self):
        return self.__class__.__name__ 
    

class ParticleResampling(object):
    def __init__(self, probability=1):
        self.probability = probability

    def __call__(self, img):
        if np.random.rand() > self.probability:
            return img  # Return the image without any change

        intensity_img = img[:,:,0]
        fraction = self._calculate_fraction(intensity_img)

        hits_array = self._image_to_hits_array(intensity_img)
        augmented_hits = self._resample_hits(hits_array, fraction)
        aug_intensity_image = self._hits_array_to_image(augmented_hits, intensity_img.shape)
        aug_image = img
        aug_image[:, :, 0] = aug_intensity_image
        return aug_image

    def _calculate_fraction(self, img):
        intensity = np.mean(img)
        if intensity < 0.05:
            return 1
        elif 0 <= intensity < 1:
            return np.random.uniform(0.7, 1)
        else:
            return np.random.uniform(0.5, 1)

    def _image_to_hits_array(self, img):
        row_indices, col_indices = np.nonzero(img)
        counts = img[row_indices, col_indices]
        hits_array = np.repeat(np.column_stack((row_indices, col_indices)), counts.astype(int), axis=0)
        return hits_array

    def _resample_hits(self, hits_array, fraction):
        num_samples = int(len(hits_array) * fraction)
        indices = np.random.choice(len(hits_array), num_samples, replace=False)
        return hits_array[indices]
    

    def _upsample_hits(self, hits_array, fraction):
        num_additional_samples = int(len(hits_array) * fraction)
        additional_samples = hits_array[np.random.choice(len(hits_array), num_additional_samples, replace=True)]
        return np.concatenate((hits_array, additional_samples))

    def _hits_array_to_image(self, hits_array, shape):
        aug_image = np.zeros(shape, dtype=int)
        row_indices, col_indices = hits_array.T
        np.add.at(aug_image, (row_indices, col_indices), 1)
        return torch.from_numpy(aug_image)    

def plot_comparison(original, augmented, title=''):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Calculate vmin and vmax across both images
    vmin = min(np.min(original), np.min(augmented))
    vmax = max(np.max(original), np.max(augmented))
    
    # Create a normalization that spans both images
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the original image
    im1 = axes[0].imshow(original, cmap='viridis', interpolation='nearest', norm=norm)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Plot the augmented image
    im2 = axes[1].imshow(augmented, cmap='viridis', interpolation='nearest', norm=norm)
    axes[1].set_title(f'Augmented (Fraction: {title})')
    axes[1].axis('off')
    
    # Create a separate axis for the colorbar
    cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])  # Adjust the position and size as needed
    
    # Create a shared colorbar
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
    cbar.set_label('Hits Count')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    create_train_indices_random()
