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


# change cuda to True if you wish to use gpu
cuda = True
images = "../1RawData/"
det = "../2ProcessedData/"
batch_size = 1
confidence = float(0.4)
nms_thesh = float(0.40)
start = 0
num_classes = 4
colors = pkl.load(open("../4Others/pallete", "rb"))
# one anchor 
#cfg_path = "../4Others/color_ball_one_anchor.cfg"

# 3 anchors
cfg_path = "../4Others/color_ball_one_anchor.cfg"
blocks = parse_cfg(cfg_path)
model = yolo_v3(blocks)
classes = load_classes('../4Others/color_ball.names')

# 416

checkpoint_path = "../4TrainingWeights/experiment/one_anchor/480_seed_427_2018-12-19_22_02_06.032829/2018-12-19_22_26_13.738629_model.pth"


# 512

checkpoint_path = "../4TrainingWeights/experiment/multiple_train/_seed_425_2018-12-19_00_32_29.569630/2018-12-19_00_35_55.883400_model.pth"


# 608

#checkpoint_path = "../4TrainingWeights/experiment/multiple_train/_seed_1218_2018-12-18_18_43_17.797231/2018-12-18_19_20_48.644387_model.pth"
#checkpoint_path = "../4TrainingWeights/experiment/input_size/700_to_750_imgs_seed_429_2018-12-09_05_01_14.012722/2018-12-09_05_16_05.907534_model.pth"



checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
for params in model.parameters():
    params.requires_grad = False
if cuda:
    model = model.cuda()
#    torch.set_num_threads(8)

## 608
#model.net["height"] = 608
#model.net["width"] = 608
#
# 512
model.net["height"] = 448
model.net["width"] = 448

## 416
#model.net["height"] = 416
#model.net["width"] = 416

inp_dim = model.net["height"]
transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
# yolo v3 down size the imput images 32 strides, therefore the input needs to
# be a multiplier of 32 and > 32
assert inp_dim % 32 == 0
assert inp_dim > 32


# Set the model in evaluation mode
# model = model.eval()
#
# Detection phase

# videofile = args.videofile #or path to the video file.

# cap = cv2.VideoCapture(videofile)


cap = cv2.VideoCapture(0)
# 480 p 
cap.set(3, 800)

# 720 p 
#cap.set(3, 1280)

cap.get(3)
cap.get(4)
fps_list = []
add = False
count_time = 10
count_start_time = 100000
assert cap.isOpened(), 'Cannot capture source'
start = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        img = letterbox_image(frame, (inp_dim, inp_dim))
        # change from w h c to c h w for pytorch imput
        img = img[:,:,::-1].transpose((2, 0, 1)).copy()
        img = torch.FloatTensor(img)
        img = transform(img).unsqueeze(0)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
        if cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = model(img, cuda)
        output = filter_results(output, confidence, num_classes, nms_thesh)

        if type(output) == int:
            cv2.imshow("frame", frame)
            fps = (1 / (time.time() - start))
#            print(f"FPS of the video is {fps:5.4f}")
            if add:
                fps_list.append(fps)
                if (time.time() - count_start_time) > count_time:
#                    print(f"avg_fps: {np.mean(fps_list):5.4f}")
                    break
            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'):
                count_start_time = time.time()
                fps_list.append(fps)
                add = True
            start = time.time()
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
                break
        if key == ord('b'):
            count_start_time = time.time()
            fps_list.append(fps)
            add = True
        start = time.time()
        if key == 27:
            break
    else:
        break
