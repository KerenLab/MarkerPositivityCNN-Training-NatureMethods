import os
from pprint import pprint
from re import I, T
import time
from typing import Dict, List, Tuple
from xmlrpc.client import TRANSPORT_ERROR
import cv2
from matplotlib import contour, pyplot as plt

import numpy as np
from sklearn import base
from sklearn.metrics import mean_absolute_error, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from evaluation.perf_calculator import PerfCalculator
from data_tools.channels_data import MarkerType, get_channel_list, get_channel_list_from_project_mantis_dir, get_marker_type_encoding
from data_tools.dataset_utils import CellIdentifier, get_segmentation_image_of_fov, get_wexac_base_path
from data_tools.datasets import FOVDatasetByLabelDict, FOVDatasetByMantisFolder
from data_tools.torch_models import ResNet18FeatureExtractor, ResNet18CellWithSegAndExpressionType, ResNet18CellWithSegFeatureExtractor, model_efficient
import pandas as pd
from sys import platform
from data_tools.datasets import parse_label_csv_files
import psutil
 


def profile(path_to_mantis_dir: str,
            path_to_model_weight_file:str,
            label_csv_path_list:str,
            path_to_csv_out: str,
            dataset_is_seg:bool = True,
            use_marker_expression_as_input:bool=False):
    
    print('Beginning profiling!')
    fov_dir_path_list =  [os.path.join(path_to_mantis_dir, d) for d in os.listdir(path_to_mantis_dir) if os.path.isdir(os.path.join(path_to_mantis_dir, d))]
    fov_dir_path_list.sort()
    label_dict = None
    fov_dir_path_list = fov_dir_path_list[:15]
    proj_name = os.path.basename(label_csv_path_list[0]).split('_')
    proj_name = proj_name[0] + '_' + proj_name[1]

    channel_list = get_channel_list_from_project_mantis_dir(path_to_mantis_dir, get_only_tif_files=False)
    print('RAM Used (MB) before:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    print(f'Building datasets...')
    dataset = FOVDatasetByMantisFolder(manits_img_dir_path_list=fov_dir_path_list,
                                        channel_list=channel_list,
                                          labels_dict=label_dict,
                                            seg_as_channels=dataset_is_seg,
                                            add_augmentations=False,
                                            get_metadata=True,
                                            cache_images=True)
    
    print('RAM Used (MB) after dataset creation:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)


    if dataset_is_seg:
        if use_marker_expression_as_input:
            model = ResNet18CellWithSegAndExpressionType(num_classes=1, feature_dim=32, dropout_rate = 0.5)
        else:
            model = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)
    else:
        model = ResNet18FeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    model = model.to(device)

    print(f'Using weight file: {os.path.basename(path_to_model_weight_file)}')
    state_curr = torch.load(path_to_model_weight_file, map_location=device)

    model.load_state_dict(state_curr)

    print(f'Starting evaluation on {proj_name}, over {len(fov_dir_path_list)} FOVs. Model uses segmentation? {dataset_is_seg}')


    model.eval()

    def profile_dataloader(dataset, bs, nw):
        data_loader = DataLoader(dataset, batch_size=bs, num_workers=nw, shuffle=True, persistent_workers=True)
        r1  = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        it = iter(data_loader)

        tic = time.time()
        images, labels, cell_data, fov_ind, tt = next(it)
        tt = tt.numpy()
        toc = time.time()
        r2 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        print('diff ram used:', r2- r1)
        print('Time to load data with batch size = {}, num worker = {} was {:.2f}'.format(bs, nw, toc-tic))
        tic = time.time()
        if use_marker_expression_as_input:
            _, _, _, expression = cell_data
            outputs, _ = model(images.float().to(device), expression.reshape(-1,1).float().to(device))
        else:
            if model.__class__.__name__ == 'EfficientNet':
                outputs = model((images.float().to(device)))
            else:
                outputs, _ = model((images.float().to(device)))

        toc = time.time()
        print('Time to perform inference with batch size = {}, num worker = {} was {:.2f}'.format(bs, nw, toc-tic))
        print(f'Number of unique FOVs requested was {len(np.unique(fov_ind))}')
        print('# new images = {}, Mean time to load new images was {:.2f} msec'.format(np.sum(tt>1e-4), np.mean(tt[tt>1e-4]) * 1000))
        print('# cached images used = {}, Mean time to load cached images was {:.2f} msec'.format(np.sum(tt<1e-4), np.mean(tt[tt<1e-4]) * 1000))

    profile_dataloader(dataset, 512, 1)
    profile_dataloader(dataset, 512, 2)
    profile_dataloader(dataset, 512, 8)


def eval_model(
    model,
    device,
    test_loader,
    th=0.5,
    ratio_to_eval=1,
    str2add="",
    *,
    verbose: bool = True,
):
    """If ``verbose`` is False, prints one validation summary line (for training loops)."""
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    total_perf_calc = PerfCalculator(f"Eval {str2add}: All markers")
    nuclear_perf_calc = PerfCalculator(f"Eval {str2add}: Nuclear markers")
    cyto_perf_calc = PerfCalculator(f"Eval {str2add}: Cytoplasmic markers")
    membranal_perf_calc = PerfCalculator(f"Eval {str2add}: Membranal markers")

    num_batch_to_eval = len(test_loader) * ratio_to_eval
    cur_batch = 0
    tot_loss = 0
    with torch.no_grad():
        for images, labels, cell_data in tqdm.tqdm(
            test_loader,
            desc="Eval batches",
            total=len(test_loader),
            leave=False,
            disable=not verbose,
        ):
            if cur_batch > num_batch_to_eval:
                break
            cur_batch +=1
            _, _, _, expression = cell_data
            if model.__class__.__name__ == 'EfficientNet':
                outputs = model((images.float().to(device)))
            else:
                outputs, _ = model((images.float().to(device)))
            loss = criterion(outputs, labels.float().unsqueeze(1).to(device))
            tot_loss += loss.item()

            indices_membranal = torch.nonzero((expression == MarkerType.Membranal) & (labels != -1), as_tuple=False).squeeze()
            indices_cyto = torch.nonzero((expression == MarkerType.Cytoplasmic) & (labels != -1), as_tuple=False).squeeze()
            indices_nuclear = torch.nonzero((expression == MarkerType.Nuclear) & (labels != -1), as_tuple=False).squeeze()

            predicted = torch.sigmoid(outputs) >= th
            inds_to_get = torch.nonzero(labels != -1)
            if len(labels[inds_to_get].size()) == 0:
                continue
            labels = labels.to(device)
            total_perf_calc(predicted[inds_to_get], labels[inds_to_get])

            if (indices_nuclear.dim() != 0) and (len(indices_nuclear) > 0):
                nuclear_perf_calc(predicted[indices_nuclear], labels[indices_nuclear])
            if (indices_cyto.dim() != 0) and (len(indices_cyto)) > 0:
                cyto_perf_calc(predicted[indices_cyto], labels[indices_cyto])
            if (indices_membranal.dim() != 0) and (len(indices_membranal)) > 0:
                membranal_perf_calc(predicted[indices_membranal], labels[indices_membranal])


    mean_loss = tot_loss / len(test_loader)
    if verbose:
        total_perf_calc.get_perf(print_results=True)
        nuclear_perf_calc.get_perf(print_results=True)
        cyto_perf_calc.get_perf(print_results=True)
        membranal_perf_calc.get_perf(print_results=True)
    else:
        acc, _, _, f1 = total_perf_calc.get_perf(print_results=False)
        nuclear_perf_calc.get_perf(print_results=False)
        cyto_perf_calc.get_perf(print_results=False)
        membranal_perf_calc.get_perf(print_results=False)
        print(f"  val: loss={mean_loss:.4f} | all-markers acc={acc}% F1={f1}%")

    return mean_loss


def eval_model_from_dirs(path_to_mantis_dirs: List[str],
                        label_csv_path_list: List[str],
                        path_to_model_weight_file:str,
                        dataset_is_seg:bool = True, 
                        use_marker_expression_as_input: bool=False):
    
    fov_dir_path_list = list()
    for i, path_to_mantis_dir in enumerate(path_to_mantis_dirs):
        fov_dir_path_list_proj =  [os.path.join(path_to_mantis_dir, d) for d in os.listdir(path_to_mantis_dir) if os.path.isdir(os.path.join(path_to_mantis_dir, d))]
        fov_dir_path_list.extend(list(fov_dir_path_list_proj)) 

    fov_dir_path_list.sort()
    label_dict, weight_list, channel_list_all = parse_label_csv_files(label_csv_path_list, fov_dir_path_list)

    print(f'Building datasets...')
    if dataset_is_seg:
        dataset = FOVDatasetByLabelDict(manits_img_dir_path_list=fov_dir_path_list, labels_dict=label_dict, add_augmentations=False)
    else:
        dataset = FOVDatasetByLabelDict(manits_img_dir_path_list=fov_dir_path_list, labels_dict=label_dict, add_augmentations=False, seg_as_channels=False)
    data_loader = DataLoader(dataset, batch_size=512, num_workers=32)

    if dataset_is_seg:
        if use_marker_expression_as_input:
            model = ResNet18CellWithSegAndExpressionType(num_classes=1, feature_dim=32, dropout_rate = 0.5)
        else:
            model = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)

    else:
        model = ResNet18FeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    model = model.to(device)

    state_curr = torch.load(path_to_model_weight_file, map_location=device)

    model.load_state_dict(state_curr)

    print(f'Starting evaluation on {len(fov_dir_path_list)} FOVs')
    print(f'Model uses segmentation? {dataset_is_seg}')
    print(f'Model uses marker expression type? {use_marker_expression_as_input}')

    model.eval()

    test_loss = eval_model(model, device, data_loader)
    print(f'Test loss: {test_loss}%')


def compare_models(path_to_w1_is_seg, path_to_w2_not_seg, path_to_mantis_dirs: List[str], label_csv_path_list: List[str]):
    fov_dir_path_list = list()
    proj_name = os.path.basename(path_to_mantis_dirs[0].replace('MantisProject', ''))
    for i, path_to_mantis_dir in enumerate(path_to_mantis_dirs):
        fov_dir_path_list_proj =  [os.path.join(path_to_mantis_dir, d) for d in os.listdir(path_to_mantis_dir) if os.path.isdir(os.path.join(path_to_mantis_dir, d))]
        fov_dir_path_list.extend(list(fov_dir_path_list_proj)) 

    fov_dir_path_list.sort()
    ind_arr = np.arange(0, len(fov_dir_path_list), 10)
    ind_arr = list(ind_arr) + [len(fov_dir_path_list)]

    ind_1 = ind_arr[1]
    ind_2 = ind_arr[2]
    fov_dir_path_list = fov_dir_path_list[ind_1:ind_2]
    try:
        label_dict, weight_list, channel_list_all = parse_label_csv_files(label_csv_path_list, fov_dir_path_list)
    except FileNotFoundError:
        print('Could not find labels file...')
        label_dict = None

    
    print(f'Building datasets...')
    channel_list = get_channel_list_from_project_mantis_dir(path_to_mantis_dir, get_only_tif_files=True)
    dataset1 = FOVDatasetByMantisFolder(manits_img_dir_path_list=fov_dir_path_list,
                                    channel_list=channel_list,
                                        labels_dict=label_dict,
                                        seg_as_channels=True,
                                        add_augmentations=False,
                                        get_metadata=True)
    
    dataset2 = FOVDatasetByMantisFolder(manits_img_dir_path_list=fov_dir_path_list,
                                    channel_list=channel_list,
                                        labels_dict=label_dict,
                                        seg_as_channels=False,
                                        add_augmentations=False,
                                        get_metadata=True)
    
    data_loader1 = DataLoader(dataset1, batch_size=4096, num_workers=12, shuffle=False)
    data_loader2 = DataLoader(dataset2, batch_size=4096, num_workers=12, shuffle=False)

    model1 = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)
    model2 = ResNet18FeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    model1 = model1.to(device)
    model2 = model2.to(device)

    state_curr1 = torch.load(path_to_w1_is_seg, map_location=device)
    model1.load_state_dict(state_curr1)

    state_curr2 = torch.load(path_to_w2_not_seg, map_location=device)
    model2.load_state_dict(state_curr2)

    print(f'Starting evaluation on {len(fov_dir_path_list)} FOVs')

    model1.eval()
    model2.eval()
    th = 0.5
    agree_n = 0
    disagree_n = 0
    seg_is_bigger=0
    total = 0
    diff_prob = list()
    fov_list = list()
    cell_id_list = list()
    channel_dis_list = list()
    prob_list_seg = list()
    prob_list_non_seg = list()
    with torch.no_grad():
        for data1, data2 in tqdm.tqdm(zip(data_loader1, data_loader2), total=len(data_loader1)):
            images1, labels1, cell_data1 = data1
            images2, labels2, cell_data2 = data2

            outputs1, _ = model1((images1.float().to(device)))
            predicted1 = torch.sigmoid(outputs1) >= th

            outputs2, _ = model2((images2.float().to(device)))

            outputs2[torch.isclose(torch.sigmoid(outputs2),torch.tensor(0.2924), atol=1e-3)] = -10000
            predicted2 = torch.sigmoid(outputs2) >= th
            fov_name, channel_name, cell_id, cell_exp =  cell_data1
            seg_is_bigger += torch.sum(torch.sigmoid(outputs1[predicted1 != predicted2]) > torch.sigmoid(outputs2[predicted1 != predicted2]) )
            diff_prob.extend(list((torch.sigmoid(outputs1) - torch.sigmoid(outputs2)).cpu().numpy().squeeze()))
            disagree_n += torch.sum(predicted1 != predicted2)
            agree_n += torch.sum(predicted1 == predicted2)

            inds_disagree = (torch.nonzero(torch.abs(torch.sigmoid(outputs1) - torch.sigmoid(outputs2)).squeeze() > 0.25).cpu().numpy().astype(int))
            if torch.nonzero(torch.abs(torch.sigmoid(outputs1) - torch.sigmoid(outputs2)).squeeze() > 0.25).nelement() == 0:
                continue
            if len(inds_disagree) != 1:
                inds_disagree = inds_disagree.squeeze()
            fov_list.extend(list(np.array(fov_name)[inds_disagree]))
            cell_id_list.extend(list(np.array(cell_id)[inds_disagree]))
            channel_dis_list.extend(list(np.array(channel_name)[inds_disagree]))
            prob_list_seg.extend(list(torch.sigmoid(outputs1).cpu().numpy().squeeze()[inds_disagree]))
            prob_list_non_seg.extend(list(torch.sigmoid(outputs2).cpu().numpy().squeeze()[inds_disagree]))

    print(f'Agree percent: {agree_n/(agree_n+disagree_n)*100}%')
    print(f'Disagree percent: {disagree_n/(agree_n+disagree_n)*100}%')
    print(f'New network predicted higher probability in {seg_is_bigger} out of {(disagree_n)} disagreement cases')
    print(f'mean difference between prob values is {np.mean(diff_prob)}')

    df = pd.DataFrame({
    'FOV': fov_list,
    'CellID': cell_id_list,
    'Channel': channel_dis_list,
    'Prob_seg_new': prob_list_seg,
    'Prob_non_seg_old': prob_list_non_seg,
    })

    dir_to_save = os.path.join('Disagreements', proj_name)
    path_to_save = os.path.join(dir_to_save, 'disagree_table.csv')
    os.makedirs(dir_to_save, exist_ok=True)
    df.to_csv(path_to_save, index=False)

    print(f'Done! saved results to {path_to_save}')


def predict_specific_cells(model_w_path_list:List[str], model_class_list:list[str], cell_identifiers_list:List[CellIdentifier], differentiate_expr_type:bool=False):
    fov_dir_path_list = list()
    proj_name_list = list()
    ch_list = list()
    for cellIDData in cell_identifiers_list:
        if cellIDData.channel not in ch_list:
            ch_list.append(cellIDData.channel)
        fov_dir_path_list.append(os.path.join(get_wexac_base_path(), 'Collaboration', 'CellTune', 'Projects', cellIDData.proj_name, 'MantisProject' + cellIDData.proj_name, cellIDData.fov))
        if cellIDData.proj_name not in proj_name_list:
            proj_name_list.append(cellIDData.proj_name)



    print(f'Building datasets...')
    dataset_for_seg = FOVDatasetByMantisFolder(manits_img_dir_path_list=fov_dir_path_list,
                                                channel_list=ch_list,
                                                labels_dict=None,
                                                seg_as_channels=True,
                                              )
    dataset_for_non_seg = FOVDatasetByMantisFolder(manits_img_dir_path_list=fov_dir_path_list,
                                                channel_list=ch_list,
                                                labels_dict=None,
                                                seg_as_channels=False,
                                              )

    model_seg = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)

    model_non_seg = ResNet18FeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    model_seg = model_seg.to(device)
    model_non_seg = model_non_seg.to(device)

    model_list = list()
    for mode_path, model_type in zip(model_w_path_list ,model_class_list):
        state_curr = torch.load(mode_path, map_location=device)
        if model_type == 'seg':
            model = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)
        else:
            model = ResNet18FeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)

        if next(iter(state_curr.keys())).startswith('module.'):
            state_curr = {k[7:]: v for k, v in state_curr.items()}
        try:
            model.load_state_dict(state_curr)
        except RuntimeError:
            model = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5, resnet_scale=50)
            model.load_state_dict(state_curr)

        model.eval()
        model.to(device)
        model_list.append(model)
        

    if differentiate_expr_type:
        model_nuclear = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)
        model_nuclear.load_state_dict(torch.load('trained_models/Resnet18_seg_gmpn_data_as_original_nuclear_4.pth', map_location=device))
        model_nuclear.eval()
        model_nuclear.to(device)
        
        model_cytoplasmic = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)
        model_cytoplasmic.load_state_dict(torch.load('trained_models/Resnet18_seg_gmpn_data_as_original_cytoplasmic_4.pth', map_location=device))
        model_cytoplasmic.eval()
        model_cytoplasmic.to(device)
        model_memb = ResNet18CellWithSegFeatureExtractor(num_classes=1, feature_dim=32, dropout_rate = 0.5)
        model_memb.load_state_dict(torch.load('trained_models/Resnet18_seg_gmpn_data_as_original_from_start_membranal_3.pth', map_location=device))
        model_memb.eval()
        model_memb.to(device)

        model_list.extend([model_nuclear])
        model_class_list.extend(['exp_specific'])
        model_w_path_list.extend([''])

    with torch.no_grad():
        for cellIDData in cell_identifiers_list:
            cell_fov_ext = cellIDData.proj_name + '_' + cellIDData.fov
            data_seg = dataset_for_seg.get_cell_by_att(cell_fov_ext, cellIDData.channel, cellIDData.cell_id)
            data_non_seg = dataset_for_non_seg.get_cell_by_att(cell_fov_ext, cellIDData.channel, cellIDData.cell_id)
            print(f'Project name : {cellIDData.proj_name}, FOV: {cellIDData.fov}')
            print(f'Channel : {cellIDData.channel}, cellID: {cellIDData.cell_id}')
            text = ''
            for model, model_type, model_path in zip(model_list, model_class_list, model_w_path_list):

                if model_type == 'seg':
                    model_name = 'new model (seg)'
                    images, labels, cell_data = data_seg
                    _, channel, _, expression = cell_data

                elif model_type == 'exp_specific':
                    model_name = 'new model (seg)'
                    images, labels, cell_data = data_seg
                    _, channel, _, expression = cell_data
  
                    if get_marker_type_encoding(channel) == MarkerType.Nuclear:
                        model = model_nuclear
                        model_name += '_nuclear'
                    elif get_marker_type_encoding(channel) == MarkerType.Cytoplasmic:
                        model = model_cytoplasmic
                        model_name += '_cytoplasmic'
                    else:
                        model = model_memb
                        model_name += '_membranal'
                else:
                    model_name = 'Old model (non seg)'
                    images, labels, cell_data = data_non_seg

                outputs, _ = model((images.unsqueeze(0).float().to(device)))
                output_probs = torch.sigmoid(outputs)
                print(output_probs)
                text += 'For {}, probability is {:.3f}\n'.format(model_name, output_probs.squeeze().cpu().detach().numpy())
            print(text)
            print('--------------------------------------------------------------------------------------')
            from evaluation.analyze_performance import plot_single_cell

            plot_single_cell(proj_name=cellIDData.proj_name, fov = cellIDData.fov,
                                cellID=cellIDData.cell_id, marker2plot=cellIDData.channel,
                                label=labels, str_to_add_to_title=text)

    
def run_on_specific_cells():
    weight_file_name = 'Resnet18_V5_32x32_1D_GMPN_ALL_SGD_1e2_G9_dropout5_256_7.pth'
    path_to_model_weight_file_original = os.path.join(get_wexac_base_path(), 'eliovits/marker_positivity_cnn/models', weight_file_name)
    path_to_model_weight_file_list = ['trained_models/Resnet18_seg_gmpn_less_train_9.pth', path_to_model_weight_file_original]

    proj_name = 'GVHD_Feb1323'
    cell_list = [CellIdentifier(proj_name, fov='Control_01_FOV_1', cell_id=2506, channel='Mucin')]

    predict_specific_cells(model_w_path_list = path_to_model_weight_file_list,
                            model_class_list = ['seg', 'non_seg'],
                              cell_identifiers_list=cell_list, differentiate_expr_type=False)



def run_compare_pdac():
    weight_file_name = 'Resnet18_V5_32x32_1D_GMPN_ALL_SGD_1e2_G9_dropout5_256_7.pth'
    path_to_model_weight_file_original = os.path.join(get_wexac_base_path(), 'eliovits/marker_positivity_cnn/models', weight_file_name)
    path_to_model_weight_file = 'trained_models/Resnet18_seg_gmpn_data_as_original_3.pth'

    proj_name = 'MBM_Jan2524'
    path_to_manits_dir_list = [(os.path.join(get_wexac_base_path(), 'Collaboration', 'CellTune', 'Projects', proj_name, 'MantisProject' + proj_name))]
    path_to_label_csv_list = [(os.path.join(get_wexac_base_path(),  'Collaboration', 'CellTune_MarkerPositivityCNN',proj_name + '_markerPositiveTrainingTable.csv'))]
    compare_models(path_to_model_weight_file, path_to_model_weight_file_original, path_to_manits_dir_list, path_to_label_csv_list)



if __name__ == '__main__':
    run_on_specific_cells()
