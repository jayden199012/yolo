from __future__ import division

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import parse_cfg, prep_labels, my_collate, prep_params,\
 worker_init_fn
from yolo_v3 import yolo_v3
from train import run_training
from collections import ChainMap
import itertools as it
import datetime
import os


if __name__ == "__main__":
    config = {'label_csv_mame': '../1TrainData/label.csv',
              'img_txt_path': "../1TrainData/*.txt",
              'root_dir': "../1TrainData",
              'test_root_dir': "../1TestData",
              # label csv column names
              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
              'test_label_csv_mame': '../1TestData/label.csv',
              'test_img_txt_path': "../1TestData/*.txt",
              'cfg_path': "../4Others/yolo.cfg",
              'params_dir': '../4Others/params.txt'}

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    params = prep_params(config['params_dir'], config['label_csv_mame'])
    tune_params = {'seed': list(np.arange(1., 6.)),
                   'epochs': list(np.arange(20., 40., 5.)),
                   'num_anchors': [1, 3],
                   'batch_size': [2, 4, 6]}
    date_time_now = str(
            datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    compare_path = f"../5Compare/batch_size/{date_time_now}/"
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)
    index, values = zip(*tune_params.items())
    experiments_params = [dict(zip(index, v)) for v in it.product(*values)]
    for experiment_params in experiments_params:
        final_param = {**params, **experiment_params}
        best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
            map_frame = run_training(params=final_param, **config)
        map_frame.to_csv(f"{compare_path+sub_name}.csv",
                         index=True)
        time_taken_df.loc[batch_size, seed] = time.time() - x_start
        time_taken_df.to_csv(compare_path + 'time_taken.csv', index=True)
