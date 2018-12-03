from __future__ import division

import time
import torch
import os
import cv2
import pickle as pkl
from torchvision import transforms
from apscheduler.schedulers.background import BackgroundScheduler
from utilis import load_classes, parse_cfg, write, up_or_down
from utilis import filter_results, letterbox_image, action_output
# import random
from yolo_v3 import yolo_v3


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
checkpoint_path = "../4TrainingWeights/2018-11-07_23_13_38.391465/2018-11-08_02_45_20.195250_model.pth"
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


#Set the model in evaluation mode
#model = model.eval()
#
#Detection phase

#videofile = args.videofile #or path to the video file. 

#cap = cv2.VideoCapture(videofile)  


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
assert cap.isOpened(), 'Cannot capture source'
arg_list = []
started_playing = [0]
playing = []
previous_action = [-1]
distance_threshhold = .1 * inp_dim
start = time.time()
sched = BackgroundScheduler()
sched.add_job(action_output, args=[
        arg_list, previous_action, started_playing, playing
        ], trigger='interval', seconds=1.5)
sched.start()
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1920,1080))
    if ret:   
#        img = prep_image(frame, inp_dim)
        img = letterbox_image(frame, (inp_dim, inp_dim))
        img = img[:,:,::-1].transpose((2,0,1)).copy()
        img = torch.from_numpy(img).float()
        img = transform(img).float().unsqueeze(0)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
        if cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = model(img, cuda)
        output = filter_results(output, confidence, num_classes, nms_thesh)

        if type(output) == int:
            print("FPS of the video is {:5.4f}".format( 1 / (time.time() - start)))
            if started_playing[-1]:
                playing.append(0)
            start = time.time()
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                os.system("TASKKILL /F /IM vlc.exe")
                sched.shutdown()
                break
            continue

        
#        up_or_down(output, arg_list, inp_dim)
        up_or_down(output, arg_list, distance_threshhold, classes)
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
        if key == 27:
            os.system("TASKKILL /F /IM vlc.exe")
            sched.shutdown()
            break
        print("FPS of the video is {:5.2f}".format( 1 / (time.time() - start)))
        start = time.time()
    else:
        sched.shutdown()
        break     

