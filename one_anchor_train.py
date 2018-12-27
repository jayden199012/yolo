from utilis import load_classes, parse_cfg
from yolo_v3 import yolo_v3
from train import main
import random
import numpy as np
import pandas as pd
import os
import time
import torch
import logging
import datetime
from generate_anchors import set_anchors_to_model


def compare():
    date_time_now = str(
        datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    compare_path = f"../5Compare/one_anchor_input_size/{date_time_now}/"
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)
    num_anchors = 3
    conf_list = np.arange(start=0.1, stop=0.95, step=0.025)
    seed_range = list(range(424, 428))
    input_sizes = [416, 448, 480, 512, 608]
    classes = load_classes('../4Others/color_ball.names')
    cfg_path = "../4Others/color_ball_one_anchor.cfg"
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_weights("../4Weights/yolov3.weights", cust_train_zero=True)
    model.net['seed_range'] = seed_range
    label_csv_mame = '../color_balls/label.csv'
    img_txt_path = "../color_balls/*.txt"
    root_dir = "../color_balls"
    time_taken_df = pd.DataFrame(columns=list(seed_range))
    for input_size in input_sizes:
        # now parse the size into the model
        model.net['width'] = input_size
        model.net['height'] = input_size
        set_anchors_to_model(model, num_anchors, label_csv_mame, input_size,
                             input_size)
        yolo_layer = model.layer_type_dic['yolo'][0]
        model.net['anchors'] = model.module_list[yolo_layer][0].anchors
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
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
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

#   to open up the pickle configuration file
#    compare_path ='../5Compare/one_anchor_input_size_608/2018-12-18_18_43_15.712088/'
#    config_name = 'exp_config.p'
#    with open(compare_path + config_name, 'rb') as fp:
#        b = pickle.load(fp)
