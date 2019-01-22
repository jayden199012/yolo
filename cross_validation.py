from __future__ import division

import torch
from utilis import prep_params, prep_labels
from train import run_training
import itertools as it
from img_size_experiment import split_train
import shutil
import os


def cross_validation():
    config = {'label_csv_mame': '../2CvTrain/label.csv',
              'img_txt_path': "../2CvTrain/*.txt",
              'root_dir': "../2CvTrain",
              'test_label_csv_mame': '../1TestData/label.csv',
              'test_root_dir': "../1TestData",
              'test_img_txt_path': "../1TestData/*.txt",
              'valid_label_csv_mame': '../2CvValid/label.csv',
              'valid_root_dir': "../2CvValid",
              'valid_img_txt_path': "../2CvValid/*.txt",
              # label csv column names
              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
              'cfg_path': "../4Others/yolo.cfg",
              'params_dir': '../4Others/params.txt'}

    prep_label_config = {'label_csv_mame': '../1TrainData/label.csv',
                         'img_txt_path': "../1TrainData/*.txt",
                         'name_list': config['name_list']}

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # you can specify any tuning hyper-parammeters here , but do not use
    # np.arrange() as it will result an error for json.dumps()
    tune_params = {'seed': list(range(1, 6)),
                   'epochs': list(range(20, 40, 5)),
                   'batch_size': [2, 4, 6],
                   'num_anchors': [3, 1]
                   }
    index, values = zip(*tune_params.items())
    experiments_params = [dict(zip(index, v)) for v in it.product(*values)]

    # prepare label for the entire training set
    prep_labels(**prep_label_config)

    for experiment_params in experiments_params:
        cv_split_config = {'n_splits': 5,
                           'cv': True,
                           'train_size': 0.5,
                           'name_list': config['name_list'],
                           'random_state': experiment_params['seed'],
                           'train_cv_path': config['root_dir'],
                           'valid_cv_path': config['valid_root_dir'],
                           'label_name': prep_label_config['label_csv_mame']}

        for _ in split_train(**cv_split_config):
            params = prep_params(config['params_dir'],
                                 config['label_csv_mame'],
                                 experiment_params)
#            best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
#                map_frame = run_training(params=params, **config)
            print('cool')
            shutil.rmtree(config['root_dir'])
            shutil.rmtree(config['valid_root_dir'])

cross_validation()