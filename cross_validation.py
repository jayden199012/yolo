from __future__ import division
import pandas as pd
import torch
from utilis import prep_params, prep_labels
from train import run_training
import itertools as it
from img_size_experiment import split_train
import shutil


def cross_validation():
    config = {'label_csv_mame': '../2CvTrain/label.csv',
              'test_label_csv_mame': '../1TestData/label.csv',
              'test_img_txt_path': "../1TestData/*.txt",
              'valid_label_csv_mame': '../2CvValid/label.csv',
              'valid_img_txt_path': "../2CvValid/*.txt",
              # label csv column names
              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
              'cfg_path': "../4Others/yolo.cfg",
              'params_dir': '../4Others/params.txt'}
    # entire trainin label
    prep_label_config = {'label_csv_mame': '../1TrainData/label.csv',
                         'img_txt_path': "../1TrainData/*.txt",
                         'name_list': config['name_list']}
    # cv train label
    cv_train_label_config = {'label_csv_mame': config['label_csv_mame'],
                             'img_txt_path': "../2CvTrain/*.txt",
                             'name_list': config['name_list']}

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # you can specify any tuning hyper-parammeters here , but do not use
    # np.arrange() as it will result an error for json.dumps()
    tune_params = {'seed': list(range(1, 5)),
                   'batch_size': list(range(2, 7, 2)),
                   'height': [416, 448]
                   }
    index, values = zip(*tune_params.items())
    experiments_params = [dict(zip(index, v)) for v in it.product(*values)]

    # prepare label for the entire training set
    prep_labels(**prep_label_config)
    cv_results_list = []
    for experiment_params in experiments_params:
        cv_split_config = {'n_splits': 5,
                           'cv': True,
                           'train_size': 0.7,
                           'name_list': config['name_list'],
                           'random_state': experiment_params['seed'],
                           'train_cv_path': "../2CvTrain/",
                           'valid_cv_path': "../2CvValid/",
                           'label_name': prep_label_config['label_csv_mame']}

        for _ in split_train(**cv_split_config):
            prep_labels(**cv_train_label_config)
            params = prep_params(config['params_dir'],
                                 cv_train_label_config['label_csv_mame'],
                                 experiment_params)
            best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
                map_frame = run_training(params=params, **config)
            cv_results_list.append(params)
            shutil.rmtree(cv_split_config['train_cv_path'])
            shutil.rmtree(cv_split_config['valid_cv_path'])
            cv_results_df = pd.DataFrame(cv_results_list)
            cv_results_df.to_csv(f"{params['working_dir']}cv_results.csv")


if __name__ == '__main__':
    cross_validation()
