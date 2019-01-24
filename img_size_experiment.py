from __future__ import division

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from utilis import prep_params, move_images, move_images_cv
from train import run_training
import itertools as it
import datetime
import os
import time
import pandas as pd
import numpy as np


def move_not_zero(labels_df, label_name):
    labels_df = pd.read_csv(label_name)
    labels_df = pd.unique(labels_df.iloc[:, 0][labels_df.iloc[:, 1] != 0])
    return labels_df


def split_train(label_name, train_size, random_state=0, n_splits=1,
                cv=False, train_cv_path=None, valid_cv_path=None,
                name_list=None):
    # split
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=(1-train_size),
                                 random_state=random_state)
    labels_df = pd.read_csv(label_name)
    print(f"items per class in the original dataframe: \
          \n{labels_df.groupby('c').size()}")
    # get split index
    for train_index, others_index in sss.split(
            labels_df.img_name, labels_df.c):
        print(f"number of training images before removing duplicates image" +
              f"names:{np.size(train_index)}")
        print(f"number of test images before removing duplicates image" +
              f"names: {np.size(others_index)}")

        # train dataframe with possible dupliocates
        train_temp = labels_df.iloc[train_index, :]
        print(f"items per class before removing duplicates image names:" +
              f"{train_temp.groupby('c').size()}")

        # remove duplicates image names
        train_unique = pd.unique(train_temp.img_name)
        print(f"number of training images after removing duplicate image" +
              f"names:{len(train_unique)}")

        # new train dataframe with unique images names
        final_train_idx = labels_df.img_name.isin(train_unique)
        train = labels_df[final_train_idx]
        print(f"items per class with unique image names: \
              \n{train.groupby('c').size()}")
        print(f"Sampled {len(train)} images out of {len(labels_df)} images")
        if cv:
            valid = labels_df[~final_train_idx]
            move_images_cv([train, valid], [train_cv_path, valid_cv_path],
                           name_list)
            yield
        else:
            return train
    return


if __name__ == "__main__":
    date_time_now = str(
        datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    compare_path = f"../5Compare/batch_size/{date_time_now}/"
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)

    config = {'label_csv_mame': '../1TrainData/label.csv',
              'img_txt_path': "../1TrainData/*.txt",
              # label csv column names
              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
              'test_label_csv_mame': '../1TestData/label.csv',
              'test_img_txt_path': "../1TestData/*.txt",
              'cfg_path': "../4Others/yolo.cfg",
              'params_dir': '../4Others/params.txt'}

    # turn cudnn on or off depends on situation
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    to_path_list = {"../250_to_300_imgs/": 1.05/7,
                    "../400_to_450_imgs/": 1/4,
                    "../550_to_600_imgs/": 3/7,
                    "../700_to_750_imgs/": 4.7/7}

    '''
        you can specify more layered experiments here , but do not use
        np.arrange() as it will result an error for json.dumps()
    '''
    tune_params = {'seed': list(range(424, 428)),
                   }

    index, values = zip(*tune_params.items())
    experiments_params = [dict(zip(index, v)) for v in it.product(*values)]
    time_taken_df = pd.DataFrame(columns=tune_params['seed'])
    start = time.time()
    for to_path, train_size in to_path_list.items():
        file_name = to_path.strip(".").strip("/")
        for experiment_params in experiments_params:
            params = prep_params(config['params_dir'],
                                 config['label_csv_mame'],
                                 experiment_params)
            move_images(label_name=config['label_csv_mame'], to_path=to_path,
                        action_fn=split_train, train_size=train_size,
                        random_state=params['seed'])
            params['sub_name'] = f"{file_name}_seed_{params['seed']}_"
            x_start = time.time()
            best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
                map_frame = run_training(params=params, **config)
            map_frame.to_csv(f"{compare_path+params['sub_name']}.csv",
                             index=True)
            time_taken_df.loc[params[file_name],
                              params['seed']] = time.time() - x_start
            time_taken_df.to_csv(compare_path + 'time_taken.csv', index=True)
        time_taken = time.time()-start
        print(f"This experiment took {time_taken//(60*60)} hours : \
                                      {time_taken%60} minutes : \
                                      {time_taken%60} seconds!")
