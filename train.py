from __future__ import division

import time
import numpy as np
import json
import datetime
import os
import torch
import logging
import random
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import parse_cfg, prep_labels, my_collate, load_classes,\
 worker_init_fn
from evaluate import eval_score, get_map
from yolo_v3 import yolo_v3
from data import CustData, RandomCrop
from tensorboardX import SummaryWriter

# new version


def train(model, optimizer, cuda, config, train_loader, test_loader,
          conf_list, classes, iou_conf, save_txt, loop_conf=False,
          loop_epoch=20):
    config["global_step"] = 0
#    map_counter = 0

    # multi learning rate
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["steps"][0],
        gamma=config["scales"][0])
    # gpu
    if cuda:
        model = model.cuda()
    model.train()
    # Restore pretrain model
    if config["pretrain_snapshot"]:
        print("Load pretrained weights from {}".format(
                        config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        model.load_state_dict(state_dict)
    map_results_names = ["best_map", "best_ap", "best_conf",
                         "specific_conf_map", "specific_conf_ap"]
    # Start the training loop
    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        print("this is epoch :{}".format(epoch))
        if loop_epoch and epoch > loop_epoch:
            loop_conf = True
        for step, samples in enumerate(train_loader):
            if cuda:
                images, labels = samples["image"].to('cuda'), samples["label"]
            else:
                images, labels = samples["image"], samples["label"]

            start_time = time.time()
            config["global_step"] += 1

            # Forward and backward
            optimizer.zero_grad()
            losses = model(images, cuda, is_training=True, labels=labels)
            loss = losses[0]
            loss.backward()
            optimizer.step()

            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch"] / duration
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = \
                    %d loss = %.2f example/sec = %.3f lr = %.5f " %
                    (epoch, step, _loss, example_per_second, lr)
                )
                config["tensorboard_writer"].add_scalar("lr",
                                                        lr,
                                                        config["global_step"])
                config["tensorboard_writer"].add_scalar("example/sec",
                                                        example_per_second,
                                                        config["global_step"])
                for i, name in enumerate(model.losses_name):
                    value = _loss if i == 0 else losses[i]
                    config["tensorboard_writer"].add_scalar(
                                                        name,
                                                        value,
                                                        config["global_step"])

            if epoch > 0 and config["global_step"] % 60 == 0:
                model.train(False)
                print(f"test epoch number {epoch+1}")
                if (epoch+1) % 5 == 0:
                    # results consist best_map, best_ap, best_conf,
                    # specific_conf_map, specific_conf_ap
                    map_results = get_map(model, test_loader, cuda, conf_list,
                                          iou_conf, classes,
                                          train=True, loop_conf=loop_conf)
                    model.net['best_map'] = map_results[0]
                    model.net['confidence'] = map_results[2]
                    _save_checkpoint(model, model.state_dict(), config,
                                     save_txt)
                    for index, mr_name in enumerate(map_results_names):
                        try:
                            config["tensorboard_writer"].add_scalar(
                                    mr_name, map_results[index],
                                    config["global
                                           _step"])
                        except AttributeError:
                            continue

                evaluate_running_loss = eval_score(model, test_loader, cuda)
                for i, name in enumerate(model.losses_name):
                    config["tensorboard_writer"].add_scalar(
                                                      "evel_" + name,
                                                      evaluate_running_loss[i],
                                                      config["global_step"])
                model.train(True)
        lr_scheduler.step()

    # model.train(False)
    best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame = get_map(model, test_loader, cuda, conf_list, iou_conf,
                            classes, train=False, loop_conf=True)
    model.net['best_map'] = best_map
    model.net['confidence'] = best_conf
    _save_checkpoint(model, model.state_dict(), config, save_txt)
    for index, mr_name in enumerate(map_results_names):
        try:
            config["tensorboard_writer"].add_scalar(
                    mr_name, map_results[index],
                    config["global_step"])
        except AttributeError:
            continue
    # model.train(True)
    logging.info("Bye~")
    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame


def _save_checkpoint(model, state_dict, config, save_txt=True):
    # global best_eval_result
    time_now = str(datetime.datetime.now()).replace(
                                   " ",  "_").replace(":",  "_")
    if save_txt:
        with open(f'{config["sub_working_dir"]}/{time_now}.txt', "w") as file:
            file.write(json.dumps(model.net))

    checkpoint_path = os.path.join(config["sub_working_dir"],
                                   time_now + "_model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def main(model, classes, conf_list, label_csv_mame, img_txt_path, root_dir,
         cuda=True, specific_conf=0.5, iou_conf=0.5, sub_name='',
         selected_cls=False, return_csv=False, save_txt=True):
    date_time_now = str(
            datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    config = model.net.copy()
    test_root_dir = "../1TestData"
    # label csv column names
    name_list = ["img_name", "c", "gx", "gy", "gw", "gh"]
    prep_labels(img_txt_path, name_list, label_csv_mame,
                selected_cls=selected_cls)
    test_label_csv_mame = '../1TestData/label.csv'
    test_img_txt_path = "../1TestData/*.txt"
    prep_labels(test_img_txt_path, name_list, test_label_csv_mame,
                selected_cls=selected_cls)
    optimizer = optim.SGD(model.module_list.parameters(),
                          lr=config["learning_rate"],
                          momentum=config["momentum"],
                          weight_decay=config["decay"])
    pre_trans = RandomCrop(jitter=config['rand_crop'],
                           inp_dim=model.net["height"])
    train_transform = transforms.Compose(
            [transforms.ColorJitter(
                                    brightness=model.net["exposure"],
                                    saturation=model.net["saturation"]
                                    ),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
             ])

    test_transform = transforms.Compose(
            [transforms.Resize(model.net["height"]), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    train_data = CustData(label_csv_mame, root_dir,
                          pre_trans=pre_trans,
                          transform=train_transform)
    test_data = CustData(test_label_csv_mame, test_root_dir,
                         transform=test_transform)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=model.net["batch"],
                              collate_fn=my_collate,
                              num_workers=6, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_data, shuffle=False,
                             batch_size=model.net["batch"],
                             collate_fn=my_collate,
                             num_workers=6, worker_init_fn=worker_init_fn)
    # create working if necessary
    if not os.path.exists(config["working_dir"]):
        os.makedirs(config["working_dir"])

    # Create sub_working_dir
    sub_working_dir = os.path.join(config["working_dir"] + sub_name +
                                   date_time_now)
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={} \
                 '".format(sub_working_dir))

    # Start training
    best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame = train(model, optimizer, cuda, config, train_loader,
                          test_loader, conf_list, classes, iou_conf, save_txt)
    if return_csv:
        map_frame.to_csv(f"{config['sub_working_dir']}_final_performance.csv",
                         index=True)
    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")
    label_csv_mame = '../1TrainData/label.csv'
    img_txt_path = "../1TrainData/*.txt"
    root_dir = "../1TrainData"
    classes = load_classes('../4Others/color_ball.names')
    conf_list = np.arange(start=0.2, stop=0.95, step=0.025)
    cfg_path = "../4Others/yolo.cfg"
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_weights("../4Weights/yolov3.weights", cust_train_zero=True)
    main(model, classes, conf_list, label_csv_mame=label_csv_mame,
         img_txt_path=img_txt_path, root_dir=root_dir)
    
a = model.layer_type_dic
model_dict = model.state_dict()
model_dict['module_list.105']
model
from collections import ChainMap
a = blocks[0]
b = { 'max_batches': [12, 50],
      'policy': ['steps', 'asdas'],
      'steps': [1500, 3000, 6000]}
c = ChainMap(b, a)
c['max_batches']
def abc(**kwargs):
    for index, key in kwargs.items():
        print(f"index : {index}, key: {key}")
b['max_batches'] = 300
abc(**c)
from itertools import product
k , v = zip(*b.items())
new_items = [dict(zip(k, v_)) for v_ in product(*v)]
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
params = {'type': 'net',
          'batch': 8,
          'width': 512,
          'height': 512,
          'channels': 3,
          'momentum': 0.9,
          'decay': 0.0005,
          'angle': '0',
          'saturation': 0.25,
          'exposure': 0.25,
          'hue': 0.1,
          'rand_crop': 0.2,
          'learning_rate': 0.002,
          'steps': 50,
          'scales': 0.5,
          'epochs': 35,
          'pretrain_snapshot':
                  '../4TrainingWeights/2018-12-27_06_12_10.754957_model.pth',
          'working_dir': '../4TrainingWeights/experiment/conf_loss/',
          'num_classes': 4,
          'num_anchors': 1,
          'anchors': anchors,
          'lambda_coord': 3,
          'ignore_threshold': .7,
          'conf_lambda': 3,
          'lambda_noobj': 0.5}

tune_param = { 'max_batches': [12, 50],
              'policy': ['steps', 'asdas'],
              'steps': [1500, 3000, 6000]}
m = yolo_v3(params, blocks)
final_param = ChainMap(tune_param, params)
final_param['max_batches']
a = torch.FloatTensor(1,3,416,416).cuda()
m.cuda()
m(a, True)
s =m.layer_type_dic
m.module_list[-2][0].__code__()
blocks[81]
anchors = generate_anchor(label_csv_mame, 512, 512,
                              num_clusters=1*3)
anchors.shape
anchors[[1,2,3]]
a =([[ 26.3030303 ,  26.18181818],
       [ 32.69074074,  32.75555556],
       [ 38.54727669,  38.22309368],
       [ 44.04885737,  43.74940898],
       [ 53.535097  ,  51.85608466],
       [ 62.52395515,  61.34012912],
       [ 73.77478489,  71.93415638],
       [ 91.44729345,  90.33903134],
       [132.78787879, 131.15824916]])
a.shape
