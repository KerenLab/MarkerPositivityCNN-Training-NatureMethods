import glob
import os
from pprint import pprint
from typing import List, Union

import pandas as pd

from data_tools.dataset_utils import get_mantis_dir_path_from_proj_name, get_wexac_base_path
from enum import IntEnum

class MarkerType(IntEnum):
    Nuclear = 1
    Cytoplasmic = 2
    Membranal = 3


channel_list_gvhd = ['Calprotectin',\
                    'CD103',\
                    'CD14',\
                    'CD15',\
                    'CD163',\
                    'CD20',\
                    'CD206',\
                    'CD209',\
                    'CD3',\
                    'CD31',\
                    'CD38',\
                    'CD4',\
                    'CD45',\
                    'CD45RA',\
                    'CD45RO',\
                    'CD56',\
                    'CD68',\
                    'CD69',\
                    'CD8',\
                    'CHGA',\
                    'Collagen',\
                    'dsDNA',\
                    'ECAD',\
                    'FOXP3',\
                    'GZMB',\
                    'HLA1',\
                    'HLADRDPDQ',\
                    'IDO1',\
                    'IgA',\
                    'Keratin',\
                    'Ki67',\
                    'Lysozyme',\
                    'Mucin',\
                    'NAKATPASE',\
                    'PD1',\
                    'PDL1',\
                    'SMA',\
                    'TCF',\
                    'Tryptase',\
                    'VIM']

channel_list_mbm = ['BCAT',
                    'Calprotectin',
                    'cCasp3',
                    'CD11c',
                    'CD14',
                    'CD163',
                    'CD20',
                    'CD206',
                    'CD3',
                    'CD31',
                    'CD4',
                    'CD45',
                    'CD45RO',
                    'CD56',
                    'CD68',
                    'CD8',
                    'Collagen',
                    'dsDNA',
                    'FOXP3',
                    'GFAP',
                    'gH2AX',
                    'GLUT1',
                    'GZMB',
                    'HLA1',
                    'HLADRDPDQ',
                    'IBA1',
                    'IDH1',
                    'IDO1',
                    'IFNG',
                    'Keratin',
                    'Ki67',
                    'LAG3',
                    'MelanA',
                    'PD1',
                    'PDL1',
                    'SMA',
                    'SOX10',
                    'TCF',
                    'TIM3',
                    'Tox']

channel_list_pdac = ['Amylase', 'Calprotectin', 'cCasp3', 'CD14', 'CD15', 'CD20', 'CD3', 'CD31', 'CD4', 'CD45', 'CD8', 'COL1A1', 'CXCL5', 'dsDNA', 'ECAD', 'FAP', 'FASN', 'FOXP3', 'GATA6', 'gH2AX', 'GLUT1', 'HIF1a', 'HLA1', 'HLADRDPDQ', 'IDO1', 'IL6', 'Keratin', 'Ki67', 'KRT5', 'LDHA', 'MCT1', 'MMP7', 'NaKATPase', 'P53', 'PD1', 'PDL1', 'pNRF2', 'SMA', 'SYP', 'VIM']


def get_channel_list_from_project_mantis_dir(path_to_project_dir, get_only_tif_files:bool):
    if os.sep not in path_to_project_dir:
        path_to_project_dir = get_mantis_dir_path_from_proj_name(path_to_project_dir)

    fov_dir_path_list =  [os.path.join(path_to_project_dir, d) for d in os.listdir(path_to_project_dir) if os.path.isdir(os.path.join(path_to_project_dir, d))]
    return get_channel_list(fov_dir_path_list[1], get_only_tif_files)


def get_channel_list(example_tif_dir, get_only_tif_files):
    if get_only_tif_files:
        tif_images = glob.glob(os.path.join(example_tif_dir, '*.tif'))
    else:
        tif_images = glob.glob(os.path.join(example_tif_dir, '*.tif*'))

    marker_names = [os.path.basename(image).split('.')[0] for image in tif_images]
    marker_names = [marker for marker in marker_names if 'segmentation' not in marker]
    return marker_names



        
def get_marker_type_encoding(marker_input: Union[str, List[str], pd.Series], output_norm_encoding:bool=False):
    def process_marker(marker_name):
        if marker_name.upper() in ['FOXP3', 'SOX10', 'GATA6', 'PNRF2', 'KI67', 'P53', 'DSDNA', 'GH2AX', 'HIF1A', 'TCF', 'TOX', 'HH3']:
            # Nuclear marker
            return MarkerType.Nuclear
        elif marker_name.upper() in ['MUCIN', 'LYSOZYME', 'CHGA', 'CALPROTECTIN', 'AMYLASE', 'LDHA', 'MMP7', 'KERATIN',
                                      'IL6', 'IDO1', 'SMA', 'SYP', 'FAP', 'FASN', 'CXCL5', 'CCASP3', 'MPO_CALP', 'VIM',
                                      'TRYPTASE', 'BCAT', 'GZMB', 'HO1', 'INOS', 'IDH1', 'PDL1']:
            # Cytoplasmatic marker
            return MarkerType.Cytoplasmic
        else:
            # Membranal marker
            return MarkerType.Membranal
    
    if isinstance(marker_input, str):
        marker_types = process_marker(marker_input)

    elif isinstance(marker_input, list) or isinstance(marker_input, pd.Series):
        marker_types = pd.Series([process_marker(marker_name) for marker_name in marker_input], index=marker_input.index)
    else:
        raise ValueError("Input must be a string or a list of strings or a pd series.")
    
    if output_norm_encoding:
        if isinstance(marker_type, list):
            marker_type = [marker.value / 3 for marker in marker_type]
        else:
            marker_type = marker_type.value / 3
    
    return marker_types




if __name__ == '__main__':
    # path_to_img_dir = r'Y:\Collaboration\CellTune\Projects\Melanoma_Sept1022\MantisProjectMelanoma_Sept1022\Slide05_Point014'
    # tif_images = glob.glob(os.path.join(path_to_img_dir, '*.tif'))
    # # print(tif_images)
    # marker_names = [os.path.basename(image).split('.')[0] for image in tif_images]
    # assert len(marker_names) == 40
    # pprint(marker_names)
    a = MarkerType.Membranal
    print(type(a) == MarkerType)