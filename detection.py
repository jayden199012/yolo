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
from generate_anchors import set_anchors_to_model
from evaluate import compute_map


def detection(cfg_path, batch_size, nms_thesh, confidence, classes, det, cuda,
              checkpoint, label_csv_mame, root_dir, num_classes, width, height,
              num_anchors):
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_state_dict(checkpoint)
    # model.load_weights("../4Weights/yolov3.weights")
    if cuda:
        model = model.cuda()
    for params in model.parameters():
        params.requires_grad = False
    model.net["height"] = width
    model.net["width"] = height
    inp_dim = model.net["height"]
    set_anchors_to_model(model, num_anchors, label_csv_mame, width,
                         height)
    # yolo v3 down size the imput images 32 strides, therefore the input needs
    # to be a multiplier of 32 and > 32
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
                             num_workers=0, worker_init_fn=worker_init_fn)
    for original_imgs, images, imlist, im_dim_list, labels in data_loader:
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        if cuda:
            images = images.cuda()
            im_dim_list = im_dim_list.cuda()
        prediction = model(images, cuda)
        output = filter_results(prediction, confidence, num_classes, nms_thesh)
        if type(output) != int:
            im_dim_list = torch.index_select(
                    im_dim_list, 0, output[:, 0].long())
            scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1, 1)

            # Clamp all elements in input into the range [ min, max ] and
            # return a resulting tensor:
            # it is to make sure the min max of width and height are not larger
            # or smaller than the image boundary
            output[:, [1, 3]] -= (inp_dim - scaling_factor *
                                  im_dim_list[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (inp_dim - scaling_factor *
                                  im_dim_list[:, 1].view(-1, 1))/2
            output[:, 1:5] /= scaling_factor
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]],
                                                0.0, im_dim_list[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]],
                                                0.0, im_dim_list[i, 1])
        list(map(lambda x: detection_write(x, original_imgs, classes), output))
        det_names = pd.Series(imlist).apply(
                           lambda x: "{}/{}".format(det, x.split("/")[-1]))
        list(map(cv2.imwrite, det_names, original_imgs))
        if cuda:
            torch.cuda.empty_cache()


def simgle_img_map(model, output, labels, num_classes, classes, confidence,
                   nms_thesh):
    actual_num_labels = 0
    all_detections = [[[np.array([]) for _ in range(num_classes)]]]

    # n our model if no results it outputs int 0
    if output is not 0:
        # Get predicted boxes, confidence scores and labels
        pred_boxes = output[:, 1:6].cpu().numpy()
        scores = output[:, 5].cpu().numpy()
        pred_labels = output[:, 7].cpu().numpy()
        # Order by confidence
        sort_i = np.argsort(scores)
        pred_labels = pred_labels[sort_i]
        pred_boxes = pred_boxes[sort_i]
        for c in range(num_classes):
            all_detections[0][-1][c] = pred_boxes[
                                              pred_labels == c]
    all_annotations = []
    for label_ in labels:
        all_annotations.append(
                            [np.array([]) for _ in range(num_classes)]
                                )

        if any(label_[:, -1] > 0):
            annotation_labels = label_[label_[:, -1] > 0, 0]
            _annotation_boxes = label_[label_[:, -1] > 0, 1:]
            num_labels = len(np.unique(annotation_labels))
            # Reformat to x1, y1, x2, y2 and rescale to image dim
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] -\
                _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] -\
                _annotation_boxes[:, 3] / 2
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] +\
                _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] +\
                _annotation_boxes[:, 3] / 2
            annotation_boxes *= model.net['height']

            for label in range(num_classes):
                all_annotations[-1][label] =\
                    annotation_boxes[annotation_labels == label, :]
    # if train it results consists mAP, average_precisions map_frame
    # else: mAP, average_precisions
    actual_num_labels = np.max([actual_num_labels, num_labels])
    print(f"actual_num_labels : {actual_num_labels}")
    mAP, ap = compute_map(num_classes, classes, all_detections,
                          all_annotations, conf_index=0, confidence=confidence,
                          iou_conf=nms_thesh, map_frame=None, train=True,
                          actual_num_labels=actual_num_labels)
    return mAP, ap


def get_single_img_map(cfg_path, batch_size, nms_thesh, confidence, classes,
                       det, cuda, checkpoint, label_csv_mame, root_dir,
                       num_classes, width, height, num_anchors):
    map_frame = pd.DataFrame(columns=['mAP', 'ap'])
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    model.load_state_dict(checkpoint)
    # model.load_weights("../4Weights/yolov3.weights")
    if cuda:
        model = model.cuda()
    for params in model.parameters():
        params.requires_grad = False
    width = dim
    height = dim
    model.net["height"] = width
    model.net["width"] = height
    inp_dim = model.net["height"]
    set_anchors_to_model(model, num_anchors, label_csv_mame, width,
                         height)
    # yolo v3 down size the imput images 32 strides, therefore the input needs
    # to be a multiplier of 32 and > 32
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # put to evaluation mode so that gradient wont be calculated
    model.eval()
    transform = transforms.Compose(
        [transforms.Resize(model.net["height"]), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
#    data = CustData(label_csv_mame, root_dir, transform=transform,
#                    detection_phase=True)
    data = CustData(label_csv_mame, root_dir, transform=transform,
                    detection_phase=True)
    data_loader = DataLoader(data, shuffle=False,
                             batch_size=1,
                             collate_fn=my_collate_detection,
                             num_workers=0, worker_init_fn=worker_init_fn)
#    for original_imgs, images, imlist, im_dim_list, labels in data_loader:
#        print(images)
    for original_imgs, images, imlist, im_dim_list, labels in data_loader:
        if cuda:
            images = images.cuda()
        prediction = model(images, cuda)
        output = filter_results(prediction, confidence, num_classes, nms_thesh)
        mAP, ap = simgle_img_map(model, output, labels, num_classes, classes,
                                 confidence, nms_thesh)
        map_frame.loc[imlist[-1], 'mAP'] = mAP
        map_frame.loc[imlist[-1], 'ap'] = ap
    return map_frame


def main():
    # change cuda to True if you wish to use gpu
    cuda = True
    det = "../2ProcessedData/"
    checkpoint = torch.load(checkpoint_path)
    # classes = load_classes('../4Others/coco.names')
    classes = load_classes('../4Others/color_ball.names')
    batch_size = 1
    confidence = float(0.4)
    nms_thesh = float(0.1)
    num_classes = len(classes)
    cfg_path = "../4Others/color_ball_one_anchor.cfg"
    label_csv_mame = '../1TestData/label.csv'
    root_dir = "../1TestData"
    detection(cfg_path, batch_size, nms_thesh, confidence, classes, det, cuda,
              checkpoint, label_csv_mame, root_dir, num_classes, dim, dim,
              num_anchors)
    df = get_single_img_map(cfg_path, batch_size, nms_thesh, confidence, classes,
                           det, cuda, checkpoint, label_csv_mame, root_dir,
                           num_classes, dim, dim, num_anchors)
    return df

if __name__ == '__main__':
    num_anchors = 1
    dim = 512
    checkpoint_path = "../4TrainingWeights/experiment/one_anchor/one_anchor_input_size/448_seed_424_2018-12-27_01_19_43.494652/2018-12-27_01_43_31.599667_model.pth"
    df = main()
