from __future__ import division

import time
import torch
import cv2
from torchvision import transforms
from utilis import parse_cfg, write
from utilis import filter_results, LetterBoxImage_cv, ImgToTensorCv
from yolo_v3 import yolo_v3
import numpy as np
import json


def run_webcam(cfg_path, param_path, colors, nms_thesh=False,
               confidence=False):
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
               [LetterBoxImage_cv([params['height'], params['height']]),
                ImgToTensorCv(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

    assert params['height'] % 32 == 0
    assert params['height'] > 32

    cap = cv2.VideoCapture(0)
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
                print(f"FPS of the video is {fps:5.4f}")
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
            start = time.time()
            print(f"FPS of the video is {fps:5.4f}")
            if key == 27:
                break
        else:
            break


if __name__ == '__main__':
    config = {'colors': 0,
              # model architect file
              'cfg_path': "../4Others/yolo.cfg",
              # model parameters file
              'param_path': '../4TrainingWeights/human_eye/tune/2019-02-03_01_02_32.525378/2019-02-03_02_47_27.382756.txt'}
    run_webcam(**config)
