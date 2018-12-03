from __future__ import division

import time
import os
import os.path as osp
import numpy as np
import torch
import cv2
import pickle as pkl
from utilis import load_classes, parse_cfg, get_test_input, create_module, letterbox_image, prep_image, filter_results
import random
from yolo_v3 import yolo_v3




#layer_type_dic, module_list = create_module(blocks)
# it might imporve performance
#torch.backends.cudnn.benchmark = True

# change cuda to True if you wish to use gpu
cuda = False
images = "../1RawData/"
det = "../2ProcessedData/"
batch_size = 1
confidence = float(0.7)
nms_thesh = float(0.4)
start = 0
num_classes = 80
cfg_path = "../4Others/yolov3.cfg"
blocks = parse_cfg(cfg_path)
model = yolo_v3(blocks)
model.load_weights("../4Weights/yolov3.weights")
colors = pkl.load(open("../4Others/pallete", "rb"))
if cuda:
    model = model.cuda()
for params in model.parameters():
    params.requires_grad = False
model.layer_type_dic['net_info']["height"] = 416
model.layer_type_dic['net_info']["width"] = 416
inp_dim = int(model.layer_type_dic['net_info']["height"])

# yolo v3 down size the imput images 32 strides, therefore the input needs to
# be a multiplier of 32 and > 32
assert inp_dim % 32 == 0 
assert inp_dim > 32


#Set the model in evaluation mode
model.eval()



def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    print(cls)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#Detection phase

#videofile = args.videofile #or path to the video file. 

#cap = cv2.VideoCapture(videofile)  

cap = cv2.VideoCapture(0)

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2) 
        if cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = model(img, cuda)
        output = filter_results(output, confidence, num_classes, nms_thesh)

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
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

        classes = load_classes('../4Others/coco.names')
        colors = pkl.load(open("../4Others/pallete", "rb"))

        list(map(lambda x: write(x, frame), output))
        
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     






