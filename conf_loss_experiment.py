from utilis import load_classes, parse_cfg
from yolo_v3 import yolo_v3
from train import main
import random
import numpy as np
import pandas as pd
import pickle
import os
import time
import torch
import logging
import datetime


# label_name = '../1TestData/label.csv'
def compare():
    date_time_now = str(
        datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    compare_path = f"../5Compare/conf_lambda/{date_time_now}/"
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)
    config_name = "exp_config.p"
    conf_list = np.arange(start=0.1, stop=0.95, step=0.025)
    seed_range = range(424, 428)
    conf_lambda_list = list(range(1, 6))
    classes = load_classes('../4Others/color_ball.names')
    cfg_path = "../4Others/color_ball.cfg"
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_weights("../4Weights/yolov3.weights", cust_train_zero=True)
    model.net['seed_range'] = seed_range
    model.net['conf_lambda_list'] = conf_lambda_list
    label_csv_mame = '../color_balls/label.csv'
    img_txt_path = "../color_balls/*.txt"
    root_dir = "../color_balls"

    # save the dictionary into local pickle file
    with open(compare_path + config_name, "wb") as fp:
        pickle.dump(model.net, fp, protocol=pickle.HIGHEST_PROTOCOL)
    time_taken_df = pd.DataFrame(columns=list(seed_range))
    for conf_lambda in conf_lambda_list:
        model.conf_lambda = conf_lambda
        for index, seed in enumerate(seed_range):
            model.load_weights("../4Weights/yolov3.weights",
                               cust_train_zero=True)
            x_start = time.time()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            sub_name = f"{conf_lambda}_seed_{seed}_"
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
            time_taken_df.loc[conf_lambda, seed] = time.time() - x_start
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


# to open up the pickle configuration file
# with open(compare_path + config_name, 'rb') as fp:
#    b = pickle.load(fp)
