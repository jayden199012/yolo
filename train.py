from __future__ import division

import numpy as np
import torch
import logging
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import parse_cfg, prep_labels, my_collate, prep_params,\
 worker_init_fn, LetterboxImage_cv, ImgToTensorCv
from yolo_v3 import yolo_v3
from data import CustData, RandomCrop, CustDataCV


def run_training(params, label_csv_mame, name_list, test_label_csv_mame,
                 test_img_txt_path, cfg_path, params_dir,
                 valid_label_csv_mame=False, valid_img_txt_path=False):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    # prepare test lables
    prep_labels(test_img_txt_path, name_list, test_label_csv_mame)

    # parse model architect from architect config file
    blocks = parse_cfg(cfg_path)

    # training transformation
    pre_trans = RandomCrop(jitter=params['rand_crop'],
                           inp_dim=params["height"])
    train_transform = transforms.Compose(
            [transforms.ColorJitter(
                                    brightness=params["exposure"],
                                    saturation=params["saturation"]
                                    ),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
             ])

    test_transform = transforms.Compose(
               [LetterboxImage_cv([params['height'], params['height']]),
                ImgToTensorCv(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
    train_data = CustData(label_csv_mame,
                          pre_trans=pre_trans,
                          transform=train_transform)
    test_data = CustDataCV(test_label_csv_mame,
                           transform=test_transform)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=params["batch_size"],
                              collate_fn=my_collate,
                              num_workers=params['num_workers'],
                              worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_data, shuffle=False,
                             batch_size=params["batch_size"],
                             collate_fn=my_collate,
                             num_workers=params['num_workers'],
                             worker_init_fn=worker_init_fn)

    # initiate model
    model = yolo_v3(params, blocks)

    # Initiate validation data loader if there is any
    if valid_label_csv_mame:
        valid_data = CustDataCV(valid_label_csv_mame,
                                transform=test_transform)
        valid_loader = DataLoader(valid_data, shuffle=False,
                                  batch_size=params["batch_size"],
                                  collate_fn=my_collate,
                                  num_workers=params['num_workers'],
                                  worker_init_fn=worker_init_fn)

        # Start training
        best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
            map_frame = model.fit(train_loader, valid_loader, test_loader)
    else:

        # Start training
        best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
            map_frame = model.fit(train_loader, test_loader)

    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # initiate a dicitonary that contains all directory variable
#    config = {  # training label location
#              'label_csv_mame': '../1TrainData/label.csv',
#              # label csv column names
#              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
#              'test_label_csv_mame': '../1TestData/label.csv',
#              'test_img_txt_path': "../1TestData/*.txt",
#              # model architect file
#              'cfg_path': "../4Others/yolo.cfg",
#              # model parameters file
#              'params_dir': '../4Others/params.txt'}
    config = {  # training label location
              'label_csv_mame': '../2CvTrain/label.csv',
              # label csv column names
              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
              'test_label_csv_mame': '../2CvValid/label.csv',
              'test_img_txt_path': "../2CvValid/*.txt",
              # model architect file
              'cfg_path': "../4Others/yolo.cfg",
              # model parameters file
              'params_dir': '../4Others/params.txt'}

#    prep_label_config = {'label_csv_mame': config['label_csv_mame'],
#                         'img_txt_path': "../1TrainData/*.txt",
#                         'name_list': config['name_list']}
    prep_label_config = {'label_csv_mame': config['label_csv_mame'],
                         'img_txt_path': "../2CvTrain/*.txt",
                         'name_list': config['name_list']}

    # prepare training lables
    prep_labels(**prep_label_config)
    # prepare dictionary that contains all trainign arguments
    params = prep_params(config['params_dir'], config['label_csv_mame'])
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    # start training
    run_training(params=params, **config)
