from __future__ import division

import numpy as np
import pandas as pd
import torch
import cv2
from utilis import load_classes, parse_cfg, my_collate_detection,\
 filter_results, worker_init_fn, detection_write
from yolo_v3 import yolo_v3
from data import CustData
from torchvision import transforms
from torch.utils.data import DataLoader


def detection(cfg_path, batch_size, nms_thesh, confidence, classes, det, cuda,
              checkpoint, label_csv_mame, root_dir, num_classes):
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_state_dict(checkpoint)
    #model.load_weights("../4Weights/yolov3.weights")
    if cuda:
        model = model.cuda()
    for params in model.parameters():
        params.requires_grad = False
    model.net["height"] = 416
    model.net["width"] = 416
    inp_dim = model.net["height"]
    
    # yolo v3 down size the imput images 32 strides, therefore the input needs to
    # be a multiplier of 32 and > 32
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    
    # put to evaluation mode so that gradient wont be calculated
    model.eval()
    transform = transforms.Compose(
        [transforms.Resize(model.net["height"]), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
    data = CustData(label_csv_mame, root_dir, transform=transform,
                    detection_phase=True)
    data_loader = DataLoader(data, shuffle=False,
                         batch_size=batch_size,
                         collate_fn=my_collate_detection,
                         num_workers=4, worker_init_fn=worker_init_fn)
    for original_imgs, images, imlist, im_dim_list in data_loader:
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
        if cuda:
            images = images.cuda()
            im_dim_list = im_dim_list.cuda()
        prediction = model(images, cuda)
        prediction.shape
        output = filter_results(prediction, confidence, num_classes, nms_thesh)
        if type(output) != int:
            im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
            scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
            output[:,1:5] /= scaling_factor
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        list(map(lambda x: detection_write(x, original_imgs, classes), output))
        det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(det,x.split("/")[-1]))
        list(map(cv2.imwrite, det_names, original_imgs))
        
        if cuda:
            torch.cuda.empty_cache()

def main():
    # change cuda to True if you wish to use gpu
    cuda = True
    det = "../2ProcessedData/"
    checkpoint = torch.load(checkpoint_path)
    # classes = load_classes('../4Others/coco.names')
    classes = load_classes('../4Others/color_ball.names')
    batch_size = 2
    confidence = float(0.4)
    nms_thesh = float(0.4)
    num_classes = len(classes)
    cfg_path = "../4Others/color_ball_one_anchor.cfg"
    label_csv_mame = '../1TestData/label.csv'
    root_dir = "../1TestData"
    detection(cfg_path, batch_size, nms_thesh, confidence, classes, det, cuda,
              checkpoint, label_csv_mame, root_dir, num_classes)
    
if __name__ == '__main__':
    checkpoint_path = "../4TrainingWeights/experiment/one_anchor_input_size/512_seed_424_2018-12-19_00_10_35.635470/2018-12-19_00_32_19.696023_model.pth"
    main()