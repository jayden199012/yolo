from __future__ import division

import numpy as np
import torch
import logging
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import parse_cfg, prep_labels, my_collate, prep_params,\
 worker_init_fn
from yolo_v3 import yolo_v3
from data import CustData, RandomCrop


def run_training(params, label_csv_mame, img_txt_path, root_dir, test_root_dir,
                 name_list, test_label_csv_mame, test_img_txt_path,
                 cfg_path, params_dir):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")
    # prepare training lables
    prep_labels(img_txt_path, name_list, label_csv_mame)

    # prepare test lables
    prep_labels(test_img_txt_path, name_list, test_label_csv_mame)

    blocks = parse_cfg(cfg_path)
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
            [transforms.Resize(params["height"]), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    train_data = CustData(label_csv_mame, root_dir,
                          pre_trans=pre_trans,
                          transform=train_transform)
    test_data = CustData(test_label_csv_mame, test_root_dir,
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
    model = yolo_v3(params, blocks)
    # Start training
    best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame = model.fit(train_loader, test_loader)

    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame


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
    params = prep_params(config['params_dir'], config['label_csv_mame'])
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    run_training(params=params, **config)
