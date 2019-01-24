from __future__ import division

import time
import torch
import cv2
import pickle as pkl
from torchvision import transforms
from utilis import parse_cfg, write
from utilis import filter_results, LetterboxImage_cv, ImgToTensorCv
from yolo_v3 import yolo_v3
import numpy as np
import os
import glob
import pandas as pd
import json


def run_webcam(cfg_path, param_path, colors, count_delay, nms_thesh=False,
               confidence=False, count_time=10):
    with open(param_path) as fp:
        params = json.load(fp)
    if type(params['anchors']) == list:
        params['anchors'] = np.array(params['anchors'])
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(params, blocks)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if params['cuda']:
        model = model.cuda()
    transform = transforms.Compose(
               [LetterboxImage_cv([params['height'], params['height']]),
                ImgToTensorCv(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

    assert params['height'] % 32 == 0
    assert params['height'] > 32

    cap = cv2.VideoCapture(0)
    fps_list = []
    add = False
    print(f"this is height {params['height']}")
    if params['height'] > 480:
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
    if not confidence:
        confidence = params['confidence']
    if not nms_thesh:
        nms_thesh = params['nms_thesh']
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            # we need to unsqueeze to add batch size dimension to tensor in
            # order to fit into pytorch data format
            img = transform(frame).unsqueeze(0)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
            if params['cuda']:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(img)
            output = filter_results(output, params)
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
            scaling_factor = torch.min(
                    params['height']/im_dim, 1)[0].view(-1, 1)
            output[:, [1, 3]] -= (params['height']
                                  - scaling_factor*im_dim[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (params['height']
                                  - scaling_factor*im_dim[:, 1].view(-1, 1))/2
            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(
                        output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(
                        output[i, [2, 4]], 0.0, im_dim[i, 1])

            list(map(lambda x: write(x, frame, params['classes']), output))
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


def avg_fps(rootdir, cfg_path, count_delay, colors,
            nms_thesh, cuda, count_time,
            label_csv_mame, criteria_func=False, **kwargs):
    fps_list = []
    for root, subdirs, files in os.walk(rootdir,  topdown=False):
        if len(subdirs):
            for subdir in subdirs:
                try:
                    if criteria_func(subdir, **kwargs):
                        path = f"{root}{subdir}/*.txt"
                        files = glob.glob(path)
                    else:
                        continue
                except NameError:
                    path = f"{root}{subdir}/*.txt"
                    files = glob.glob(path)
                if len(files):
                    file = files[-1]
                    print(f"this is file {file}")
                    fps_list.append(run_webcam(
                                               cfg_path=cfg_path,
                                               param_path=file,
                                               colors=colors,
                                               count_delay=count_delay,
                                               count_time=count_time))
                    print(f"this is fps_list {fps_list}")
    print(f"avg_fps: {np.mean(fps_list):5.4f}")
    return np.mean(fps_list)


# this function should be redefined depedns on your usuage
def criteria_func(subdir, criterial_string):
    '''
        This function is to give flexibilities to find specific weights by
        adding sub directory
    '''
    if subdir.split("_")[0] == criterial_string:
        print(f'this is the string {subdir.split("_")[0]}')
        return True
    else:
        print(f'this is the string {subdir.split("_")[0]}')
        return False


if __name__ == '__main__':
    '''
    action 0:
        is to loop throught a series of weights
    to find their average fps
    action 1:
        single weight fps
    action 2 (other):
        loop through weights in result csv and input fps
    '''
    action = 2
#    dim_list = [480]
    dim_list = [416, 512, 608]
    avg_mean_fps = []
    # start evaluating fps after the count_delay
    count_delay = 4
    config = {  # training label location
              'count_delay': 4,
              # training txt file location
              'count_time': 20,
              # label csv column names
              'colors': 0,
              # model architect file
              'cfg_path': "../4Others/yolo.cfg",
              # model parameters file
              'param_path': '../4Others/params.txt'}

    # the directory you wish to loop through for weights
    rootdir = '../4TrainingWeights/experiment/one_anchor/one_anchor_input_size/'

    if action == 0:
        for dim in dim_list:
            criterial_string = str(dim)
            avg_mean_fps.append(avg_fps(
                                criteria_func=criteria_func,
                                rootdir=rootdir,
                                criterial_string=criterial_string,
                                **config))
    elif action == 1:

        # confidence = 0.375
        run_webcam(**config)
    else:
        result_csv = ['../4TrainingWeights/experiment/trial/cv_results.csv']
        for csv in result_csv:
            csv_df = pd.read_csv(csv, index_col=0)
            for index, path in enumerate(csv_df['pretrain_snapshot']):
                config['param_path'] = path.replace('pth', 'txt')
                csv_df.loc[index, 'fps'] = run_webcam(**config)
                csv_df.to_csv('./fps.csv')
    print(avg_mean_fps)
