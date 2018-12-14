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


def action_fn(label_name, cat_nums):

    labels_df = pd.read_csv(label_name)
    # train dataframe with possible dupliocates

    train_temp = labels_df[labels_df.c.isin(cat_nums)]

    # remove duplicates image names
    train_unique = pd.unique(train_temp.img_name)

    # new train dataframe with unique images names
    train = labels_df[labels_df.img_name.isin(train_unique)]

    return train


# label_name = '../1TestData/label.csv'
def compare():
    date_time_now = str(
        datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    compare_path = f"../5Compare/individual_train/{date_time_now}/"
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)
    config_name = "exp_config.p"
    conf_list = np.arange(start=0.1, stop=0.95, step=0.025)
    seed_range = range(420, 425)
    classes = load_classes('../4Others/color_ball.names')
    cfg_path = "../4Others/color_ball.cfg"
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_weights("../4Weights/yolov3.weights", cust_train_zero=True)
    with open(compare_path + config_name, "wb") as fp:
        pickle.dump(model.net, fp, protocol=pickle.HIGHEST_PROTOCOL)
    time_taken_df = pd.DataFrame(columns=list(seed_range))
    for cls_index,  cls in enumerate(classes):
        to_path = f"../{cls}/"
        for index, seed in enumerate(seed_range):
            model.load_weights("../4Weights/yolov3.weights",
                               cust_train_zero=True)
            x_start = time.time()
            random.seed(seed)
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            sub_name = f"{cls}_seed_{seed}_"
            name_list = ["img_name", "c", "gx", "gy", "gw", "gh"]
            # Original label names
            label_csv_mame = '../color_balls/label.csv'
            img_txt_path = "../color_balls/*.txt"
            prep_labels(img_txt_path, name_list, label_csv_mame)
            # sub sampled label names
            sub_sample_csv_name = to_path + "label.csv"
            sub_sample_txt_path = to_path + "*.txt"
            prep_labels(sub_sample_txt_path, name_list, sub_sample_csv_name)
            # label_csv_mame = '../1TestData/label.csv'
            # img_txt_path = "../1TestData/*.txt"
            move_images(label_name=label_csv_mame, to_path=to_path,
                        action_fn=action_fn, cat_nums=[cls_index])
            best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
                map_frame = main(model,
                                 classes, conf_list,
                                 sub_sample_csv_name,
                                 sub_sample_txt_path,
                                 to_path,
                                 cuda=True,
                                 specific_conf=0.5,
                                 sub_name=sub_name,
                                 selected_cls=[str(x) for x in [cls_index]])

            map_frame.to_csv(f"{compare_path+sub_name}.csv",
                             index=True)
            shutil.rmtree(to_path)
            time_taken_df.loc[cls, seed] = time.time() - x_start
            # if you change the csv_name, pls change accordingly in the 
            # visualization part
            time_taken_df.to_csv(compare_path + 'time_taken.csv', index=True)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
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