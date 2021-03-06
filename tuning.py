from __future__ import division

import torch
from utilis import prep_params, prep_labels
from train import run_training
import itertools as it
import random
import numpy as np

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    config = {'label_csv_mame': '../1TrainData/label.csv',
              # label csv column names
              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
              'test_label_csv_mame': '../1TestData/label.csv',
              'test_img_txt_path': "../1TestData/*.txt",
              'cfg_path': "../4Others/yolo.cfg",
              'params_dir': '../4Others/params.txt'}

    prep_label_config = {'label_csv_mame': config['label_csv_mame'],
                         'img_txt_path': "../1TrainData/*.txt",
                         'name_list': config['name_list']}

#    torch.backends.cudnn.enabled = False
#    torch.backends.cudnn.benchmark = False
    # you can specify any tuning hyper-parammeters here , but do not use
    # np.arrange() as it will result an error for json.dumps()
    tune_params = {'seed': list(range(1, 3)),
                   'height': [448, 480],
                   "decay": [0.0005, 0.001],
                   "steps": [6, 10],
                   "optimizer": ['sgd', 'adam'],
                   "epochs": [40, 80]
                   }
    index, values = zip(*tune_params.items())
    experiments_params = [dict(zip(index, v)) for v in it.product(*values)]

    # prepare training lables
    prep_labels(**prep_label_config)
    for experiment_params in experiments_params:
        params = prep_params(config['params_dir'],
                             config['label_csv_mame'],
                             experiment_params)
        try:
            best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
                map_frame = run_training(params=params, **config)
        except:
            continue
