from sklearn.model_selection import StratifiedShuffleSplit
from utilis import move_images, prep_labels, load_classes, parse_cfg
from yolo_v3 import yolo_v3
from train import main
import random
import numpy as np
import pandas as pd
import pickle
import shutil
import os
import time
import torch
import logging
import datetime


def move_not_zero(labels_df, label_name):
    labels_df = pd.read_csv(label_name)
    labels_df = pd.unique(labels_df.iloc[:, 0][labels_df.iloc[:, 1] != 0])
    return labels_df


def split_train(label_name, train_size, random_state=0):
    # split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-train_size),
                                 random_state=random_state)
    labels_df = pd.read_csv(label_name)
    print(f"items per class in the original dataframe: \
          \n{labels_df.groupby('c').size()}")
    # get split index
    train_index, others_index = next(iter(
                                   sss.split(labels_df.img_name, labels_df.c)))
    print(f"number of training images before removing duplicates image names: \
          \n{np.size(train_index)}")
    print(f"number of other images before removing duplicates image names: \
          \n{np.size(others_index)}")

    # train dataframe with possible dupliocates
    train_temp = labels_df.iloc[train_index, :]
    print(f"items per class before removing duplicates image names:\
          \n{train_temp.groupby('c').size()}")

    # remove duplicates image names
    train_unique = pd.unique(train_temp.img_name)
    print(f"number of training images after removing duplicate image names: \
          \n{len(train_unique)}")

    # new train dataframe with unique images names
    train = labels_df[labels_df.img_name.isin(train_unique)]
    print(f"items per class with unique image names: \
          \n{train.groupby('c').size()}")
    print(f"Sampled {len(train)} images out of {len(labels_df)} images")
    return train


# label_name = '../1TestData/label.csv'
def compare():
    date_time_now = str(
        datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    compare_path = f"../5Compare/input_size/{date_time_now}/"
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)
    config_name = "exp_config.p"
    conf_list = np.arange(start=0.1, stop=0.95, step=0.025)
    seed_range = range(424, 428)
    input_sizes = [608, 512, 416, 320]
    classes = load_classes('../4Others/color_ball.names')
    cfg_path = "../4Others/color_ball.cfg"
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_weights("../4Weights/yolov3.weights", cust_train_zero=True)
    model.net['seed_range'] = seed_range
    label_csv_mame = '../1TrainData/label.csv'
    img_txt_path = "../1TrainData/*.txt"
    root_dir = "../1TrainData"
    model.net['width'] = input_sizes
    model.net['height'] = input_sizes
    with open(compare_path + config_name, "wb") as fp:
        pickle.dump(model.net, fp, protocol=pickle.HIGHEST_PROTOCOL)
    time_taken_df = pd.DataFrame(columns=list(seed_range))
    for input_size in input_sizes:
        model.net['width'] = input_size
        model.net['height'] = input_size
        for index, seed in enumerate(seed_range):
            model.load_weights("../4Weights/yolov3.weights",
                               cust_train_zero=True)
            x_start = time.time()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            sub_name = f"{input_size}_seed_{seed}_"
            # label_csv_mame = '../1TestData/label.csv'
            # img_txt_path = "../1TestData/*.txt"
            best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
                map_frame = main(model,
                                 classes, conf_list,
                                 label_csv_mame,
                                 img_txt_path,
                                 root_dir,
                                 cuda=True,
                                 specific_conf=0.5,
                                 sub_name=sub_name)

            map_frame.to_csv(f"{compare_path+sub_name}.csv",
                             index=True)
            time_taken_df.loc[input_size, seed] = time.time() - x_start
            time_taken_df.to_csv(compare_path + 'time_taken.csv', index=True)


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")
    start = time.time()
    compare()
    time_taken = time.time()-start
    print(f"This experiment took {time_taken//(60*60)} hours : \
                                  {time_taken%60} minutes : \
                                  {time_taken%60} seconds!")


#with open(compare_path + config_name, 'rb') as fp:
#    b = pickle.load(fp)


# code thhat used to recover data
#labels_df = pd.read_csv(label_name)
#labels_df['img_name'] = labels_df['img_name'].str.replace(
#        '1TrainData', 'color_balls')
#img_name = pd.unique(labels_df.iloc[:, 0][labels_df.iloc[:, 1] != 0])
#try_img = img_name[0]
#for images in img_name:
#            shutil.copy(images, to_path)
#
#
#for img in img_name:
#    try_df = labels_df[labels_df["img_name"] == img]
#    try_df = try_df.iloc[:,1:]
#    xx = np.array(try_df)
#    xx[:,0] = np.int_(xx[:,0])
#    xx[:,0] = xx[:,0].astype(int)
#    with open(img[:-3] + 'txt','w') as f:
#        for items in xx:
#            for index, item in enumerate(items):
#                if not index:
#                    f.write(str((int(item))) +" ")
#                else:
#                    f.write(str(item) +" ")
#            f.write("\n")
#IMG_20181106_160325


