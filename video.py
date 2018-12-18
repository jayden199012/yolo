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
confidence = float(0.5)
nms_thesh = float(0.40)
start = 0
num_classes = 4
colors = pkl.load(open("../4Others/pallete", "rb"))
#cfg_path = "../4Others/yolov3.cfg"
cfg_path = "../4Others/color_ball.cfg"
blocks = parse_cfg(cfg_path)
model = yolo_v3(blocks)
classes = load_classes('../4Others/color_ball.names')
#model.load_weights("../4Weights/yolov3.weights")
#checkpoint_path = "../4TrainingWeights/2018-11-07_02_53_39.276476/2018-11-07_04_46_47.912754_model.pth"
#checkpoint_path = "../4TrainingWeights/input_size_expriment/608_seed_427_2018-12-10_23_31_10.005616/2018-12-10_23_58_55.119565_model.pth"
#checkpoint_path = "../4TrainingWeights/input_size_expriment/512_seed_425_2018-12-10_21_00_36.422345/2018-12-10_21_23_13.597189_model.pth"
#checkpoint_path = "../4TrainingWeights/epoch_effect/700_to_750_imgs_seed_422_epoch_35_2018-12-12_10_07_37.617714/2018-12-12_10_25_49.463584_model.pth"
#checkpoint_path = "../4TrainingWeights/experiment/input_size\700_to_750_imgs_seed_429_2018-12-09_05_01_14.012722\2018-12-09_05_16_05.907534_model.pth"
checkpoint_path = "../4TrainingWeights/experiment/epoch_experiment/700_to_750_imgs_seed_424_epoch_40_2018-12-12_12_43_27.335460/2018-12-12_13_05_15.085593_model.pth"
#checkpoint_path = "../4TrainingWeights/2018-11-07_23_13_38.391465/2018-11-08_03_43_34.979783_model.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
for params in model.parameters():
    params.requires_grad = False
if cuda:
    model = model.cuda()
#    torch.set_num_threads(8)


model.net["height"] = 416
model.net["width"] = 416
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
cap.set(3, 416)
cap.set(4, 416)
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
            print(f"FPS of the video is {fps:5.4f}")
            if add:
                fps_list.append(fps)
                if (time.time() - count_start_time) > count_time:
                    print(f"avg_fps: {np.mean(fps_list):5.4f}")
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
        print(f"FPS of the video is {fps:5.4f}")
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
