from __future__ import division

import time
import numpy as np
import datetime
import os
import torch
import logging
import random
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import parse_cfg, prep_labels, my_collate, load_classes
from evaluate import eval_score, get_map
from yolo_v3 import yolo_v3
from data import CustData, RandomCrop
from tensorboardX import SummaryWriter


def train(model, optimizer, cuda, config, train_loader, test_loader,
          conf_list, classes, iou_conf, point_confidence=True):
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

    # Start the training loop
    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        print("this is epoch :{}".format(epoch))
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
                    _save_checkpoint(model.state_dict(), config)
                    # results consist best_map, best_ap, best_conf,
                    # specific_conf_map, specific_conf_ap
                    map_results = get_map(model, test_loader, cuda, conf_list,
                                          iou_conf, classes,
                                          train=point_confidence)
                    map_results_names = ["best_map", "best_ap", "best_conf",
                                         "specific_conf_map",
                                         "specific_conf_ap"]
                    for index, mr_name in enumerate(map_results_names):
                        try:
                            config["tensorboard_writer"].add_scalar(
                                    mr_name, map_results[index],
                                    config["global_step"])
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
    _save_checkpoint(model.state_dict(), config)
    best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame = get_map(model, test_loader, cuda, conf_list, iou_conf,
                            classes, train=False, loop_conf=True)
    # model.train(True)
    logging.info("Bye~")
    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame


def _save_checkpoint(state_dict, config, evaluate_func=None):
    # global best_eval_result
    checkpoint_path = os.path.join(
            config["sub_working_dir"], str(datetime.datetime.now()).replace(
                                   " ",  "_").replace(":",  "_") + "_model.pth"
                                    )
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def worker_init_fn(worker_id):
    np.random.seed(worker_id)


def main(model, classes, conf_list, label_csv_mame, img_txt_path, root_dir,
         cuda=True, specific_conf=0.5, iou_conf=0.5, sub_name='',
         selected_cls=False, return_csv=False):
    date_time_now = str(
            datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    config = model.net
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
                          test_loader, conf_list, classes, iou_conf)
    if return_csv:
        map_frame.to_csv(f"{config['sub_working_dir']}_final_performance.csv",
                         index=True)
    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame

# type(model.net["height"])


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
    cfg_path = "../4Others/color_ball.cfg"
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_weights("../4Weights/yolov3.weights", cust_train_zero=True)
    main(model, classes, conf_list, label_csv_mame=label_csv_mame,
         img_txt_path=img_txt_path, root_dir=root_dir)
