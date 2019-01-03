from __future__ import division

import time
import torch
import cv2
import pickle as pkl
from torchvision import transforms
from utilis import load_classes, parse_cfg, write
from utilis import filter_results, letterbox_image
# import random
from yolo_v3 import yolo_v3
import numpy as np
import os
import glob
import pandas as pd
from generate_anchors import set_anchors_to_model
import json


def run_webcam(cfg_path, checkpoint_path, classes, colors, count_delay,
               nms_thesh, label_csv_mame, confidence=0, cuda=True,
               count_time=10, num_anchors=1, width=0, height=0):
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    num_classes = len(classes)
    try:
        weight_txt = checkpoint_path.replace('_model.pth', 'txt')
        with open(weight_txt) as fp:
            text = json.load(fp)
            anchors = text['anchors']
            model.net["height"] = text['height']
            model.net["width"] = text['width']
            if not confidence:
                confidence = text['confidence']
        for index, layer in enumerate(model.layer_type_dic['yolo'][::-1]):
            model.module_list[layer][0].anchors = anchors[index][0]
    except FileNotFoundError:
        set_anchors_to_model(model, num_anchors, label_csv_mame, width,
                             height)
        model.net["height"] = height
        model.net["width"] = width
    for params in model.parameters():
        params.requires_grad = False
    if cuda:
        model = model.cuda()

    inp_dim = height
    transform = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])
    # yolo v3 down size the imput images 32 strides, therefore the input needs
    # to be a multiplier of 32 and > 32
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    cap = cv2.VideoCapture(0)
    fps_list = []
    add = False
    print(f"this is height {height}")
    if height > 480:
        # 720 p
        cap.set(3, 1280)

    else:
        # 480 p
        cap.set(3, 800)
    print(f"width :{cap.get(3)}")
    print(f"height :{cap.get(4)}")
    assert cap.isOpened(), 'Cannot capture source'
    start = time.time()
    count_start_time = start + count_delay
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            img = letterbox_image(frame, (inp_dim, inp_dim))
            # change from w h c to c h w for pytorch imput
            img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
            img = torch.FloatTensor(img)
            img = transform(img).unsqueeze(0)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
            if cuda:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(img, cuda)
            output = filter_results(output, confidence, num_classes, nms_thesh)
            # when there is no dection
            if type(output) == int:
                cv2.imshow("frame", frame)
                fps = (1 / (time.time() - start))
    #            print(f"FPS of the video is {fps:5.4f}")
                if add:
                    fps_list.append(fps)
                    if (time.time() - count_start_time) > count_time:
                        print(f"avg_fps: {np.mean(fps_list):5.4f}")
                    
                        cv2.destroyAllWindows()
                        cap.release()
                        return np.mean(fps_list)
                        break
                elif time.time() > count_start_time:
                    count_start_time = time.time()
                    fps_list.append(fps)
                    add = True
                start = time.time()
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            output[:,1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

    #        classes = load_classes('../4Others/coco.names')

            list(map(lambda x: write(x, frame, classes), output))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xff
            fps = 1 / (time.time() - start)
    #        print(f"FPS of the video is {fps:5.4f}")
            if add:
                fps_list.append(fps)
                if (time.time() - count_start_time) > count_time:
                    print(f"avg_fps: {np.mean(fps_list):5.4f}")
                    cv2.destroyAllWindows()
                    cap.release()
                    return np.mean(fps_list)
                    break

            elif time.time() > count_start_time:
                count_start_time = time.time()
                fps_list.append(fps)
                add = True
            start = time.time()
            if key == 27:
                break
        else:
            break


def avg_fps(rootdir, width, height, cfg_path, count_delay, classes, colors,
            nms_thesh, cuda, count_time, csv_path, num_anchors,
            label_csv_mame, criteria_func=False, **kwargs):
    fps_list = []
    csv_df = pd.read_csv(csv_path, index_col=0)
    for root, subdirs, files in os.walk(rootdir,  topdown=False):
        if len(subdirs):
            for subdir in subdirs:
                try:
                    if criteria_func(subdir, **kwargs):
                        path = f"{root}{subdir}/*.pth"
                        files = glob.glob(path)
                    else:
                        continue
                except NameError:
                    path = f"{root}{subdir}/*.pth"
                    files = glob.glob(path)
                if len(files):
                    file = files[-1]
                    print(f"this is file {file}")
                    confidence = csv_df[
                                        csv_df.weights_path == file
                                        ].confidence.values[0]
                    fps_list.append(run_webcam(width=width, height=height,
                                               cfg_path=cfg_path,
                                               checkpoint_path=file,
                                               classes=classes,
                                               colors=colors,
                                               count_delay=count_delay,
                                               confidence=confidence,
                                               nms_thesh=nms_thesh,
                                               cuda=cuda,
                                               count_time=count_time,
                                               label_csv_mame=label_csv_mame,
                                               num_anchors=num_anchors))
                    print(f"this is fps_list {fps_list}")
    print(f"avg_fps: {np.mean(fps_list):5.4f}")
    return np.mean(fps_list)


def main(rootdir, width, height, count_delay, csv_path, num_anchors,
         label_csv_mame, count_time=10,
         cuda=True,  images="../1RawData/", det="../2ProcessedData/",
         nms_thesh=0.40, criteria_func=False, **kwargs):

    colors = pkl.load(open("../4Others/pallete", "rb"))
    classes = load_classes('../4Others/color_ball.names')
    mean_fps = avg_fps(rootdir=rootdir, width=width, height=height,
                       cfg_path=cfg_path, count_delay=count_delay,
                       classes=classes, colors=colors, nms_thesh=nms_thesh,
                       cuda=cuda, count_time=count_time,
                       criteria_func=criteria_func, csv_path=csv_path,
                       label_csv_mame=label_csv_mame, num_anchors=num_anchors,
                       **kwargs)
    return mean_fps


# this function should be redefined depedns on your usuage
def criteria_func(subdir, criterial_string):
    if subdir.split("_")[0] == criterial_string:
        print(f'this is the string {subdir.split("_")[0]}')
        return True
    else:
        print(f'this is the string {subdir.split("_")[0]}')
        return False


if __name__ == '__main__':
    '''
    action 1:
        is to loop throught a series of weights (with different seeds)
    to find their average fps
    action 2:
        single weight fps
    action 3:
        loop through weights in result csv and input fps
    '''
    action = 2
    num_anchors = 1
#    dim_list = [480]
    dim_list = [416, 512, 608]
    avg_mean_fps = []
    count_delay = 4
    nms_thesh = 0.4
    label_csv_mame = '../color_balls/label.csv'
    rootdir = '../4TrainingWeights/experiment/one_anchor/one_anchor_input_size/'
#    rootdir = '../4TrainingWeights/experiment/input_size/'
    csv_path = "../5Compare/one_anchor/top__results.csv"
    # one anchors
    cfg_path = "../4Others/color_ball_one_anchor.cfg"
    # three anchor
#    cfg_path = "../4Others/color_ball.cfg"
    if action == 0:
        for dim in dim_list:
            criterial_string = str(dim)
            avg_mean_fps.append(main(rootdir, dim, dim, count_delay,
                                csv_path=csv_path, count_time=20, cuda=True,
                                criteria_func=criteria_func,
                                criterial_string=criterial_string,
                                label_csv_mame=label_csv_mame,
                                num_anchors=num_anchors))
    elif action == 1:
        checkpoint_path = '../4TrainingWeights/experiment/one_anchor_input_size/480_seed_425_2018-12-27_03_18_42.265497/2018-12-27_03_43_00.009526_model.pth'
        colors = pkl.load(open("../4Others/pallete", "rb"))
        classes = load_classes('../4Others/color_ball.names')
        confidence = 0.375
        run_webcam(cfg_path, checkpoint_path,
                   classes, colors, count_delay, confidence, nms_thesh,
                   label_csv_mame, cuda=True, count_time=10, num_anchors=1)
    else:
        result_csv = ['../5Compare/one_anchor/top__results.csv',
                      '../4TrainingWeights/experiment/one_anchor/top__results.csv']
        for csv in result_csv:
            csv_df = pd.read_csv(csv, index_col=0)
            for index, path in enumerate(csv_df['weights_path']):
                confidence = csv_df.loc[index, 'confidence']
                csv_df.loc[index, 'fps'] = run_webcam(cfg_path, checkpoint_path,
                              classes, colors, count_delay, confidence, nms_thesh,
                            label_csv_mame, cuda=True, count_time=10, num_anchors=1)
    print(avg_mean_fps)
