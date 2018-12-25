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


def run_webcam(width, height, cfg_path, checkpoint_path, classes, colors,
               count_delay, confidence, nms_thesh, cuda=True, count_time=10):
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    num_classes = len(classes)
    for params in model.parameters():
        params.requires_grad = False
    if cuda:
        model = model.cuda()
    model.net["height"] = height
    model.net["width"] = width
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
    if height > 480:
        # 720 p
        cap.set(4, 720)

    else:
        # 480 p
        cap.set(4, 480)
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
            nms_thesh, cuda, count_time, csv_path, criteria_func=False,
            **kwargs):
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
                except:
                    path = f"{root}{subdir}/*.pth"
                    files = glob.glob(path)
                if len(files):
                    file = files[-1].replace("\\", "/")
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
                                               count_time=count_time))
                    print(f"this is fps_list {fps_list}")
    print(f"avg_fps: {np.mean(fps_list):5.4f}")
    return np.mean(fps_list)


def main(width, height, count_delay, csv_path, count_time=10,
         cuda=True,  images="../1RawData/", det="../2ProcessedData/",
         batch_size=1, nms_thesh=float(0.40), criteria_func=False, **kwargs):

    colors = pkl.load(open("../4Others/pallete", "rb"))
    classes = load_classes('../4Others/color_ball.names')
    # 3 anchors
    cfg_path = "../4Others/color_ball_one_anchor.cfg"
    # one anchor
    # cfg_path = "../4Others/color_ball_one_anchor.cfg"
    avg_fps(rootdir=rootdir, width=width, height=height, cfg_path=cfg_path,
            count_delay=count_delay, classes=classes, colors=colors,
            nms_thesh=nms_thesh, cuda=cuda, count_time=count_time,
            criteria_func=criteria_func, csv_path=csv_path, **kwargs)


if __name__ == '__main__':
    dim = 512
    count_delay = 2
    rootdir = '../4TrainingWeights/experiment/one_anchor_input_size_512/'
    csv_path = "../5Compare/top__results.csv"
    main(dim, dim, count_delay, csv_path=csv_path, count_time=20, cuda=True,
         images="../1RawData/", det="../2ProcessedData/", batch_size=1,
         confidence=float(0.4), nms_thesh=float(0.40))



