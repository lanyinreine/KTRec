import os

import torch
from longling import path_append, abs_current_dir

from KTModel.Configure import get_exp_configure
from KTModel.utils import load_model
from ..KSS.utils import load_environment_parameters as origin_load

# def load_parameters(directory=None, dataset_name='Assist2009'):
#     if directory is None:
#
#         directory = path_append(abs_current_dir(__file__), os.path.join('meta_data'))
#         # abs绝对路径 获取当前运行脚本的绝对路径 __file__本身就是绝对路径
#     environment_parameters = origin_load(os.path.join(directory, dataset_name))
#
#     return environment_parameters

Model_trained = '../KTRec/SavedModels/'


def load_d_agent(model_name, dataset_name, skill_num, device, with_label=True):
    model_parameters = get_exp_configure(model_name)
    model_parameters.update({'feat_nums': skill_num, 'model': model_name, 'without_label': not with_label})
    model = load_model(model_parameters).to(device)
    # model_folder = path_append(abs_current_dir(__file__), os.path.join('meta_data'))
    # model_path = os.path.join(model_folder, f'{model_name}_{dataset_name}_y')
    model_path = os.path.join(Model_trained, f'{model_name}_{dataset_name}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
