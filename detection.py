from __future__ import division

import numpy as np
import pandas as pd
import torch
import cv2
from utilis import parse_cfg, my_collate_detection, ImgToTensorCv,\
 filter_results, worker_init_fn, detection_write, LetterBoxImage_cv,\
 prep_labels, prep_params
from yolo_v3 import yolo_v3
from data import CustDataCV
from torchvision import transforms
from torch.utils.data import DataLoader
from evaluate import compute_map


def detection(cfg_path, det, label_csv_mame, params):
    assert params["height"] % 32 == 0
    assert params["height"] > 32
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(params, blocks)
    # model.load_weights("../4Weights/yolov3.weights")
    if params['cuda']:
        model = model.cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    # yolo v3 down size the imput images 32 strides, therefore the input needs
    # to be a multiplier of 32 and > 32

    # put to evaluation mode so that gradient wont be calculated
    model.eval()
    transform = transforms.Compose(
            [LetterBoxImage_cv([params['height'], params['height']]),
             ImgToTensorCv(), transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])

    data = CustDataCV(label_csv_mame, transform=transform,
                      detection_phase=True)
    data_loader = DataLoader(data, shuffle=False,
                             batch_size=model.params['batch_size'],
                             collate_fn=my_collate_detection,
                             num_workers=0, worker_init_fn=worker_init_fn)
    for original_imgs, images, imlist, im_dim_list, labels in data_loader:
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        if model.params['cuda']:
            images = images.cuda()
            im_dim_list = im_dim_list.cuda()
        prediction = model(images)
        model.params['num_classes']
        output = filter_results(prediction, params)
        if type(output) != int:
            im_dim_list = torch.index_select(
                    im_dim_list, 0, output[:, 0].long())
            scaling_factor = torch.min(params["height"]/im_dim_list,
                                       1)[0].view(-1, 1)

            # Clamp all elements in input into the range [ min, max ] and
            # return a resulting tensor:
            # it is to make sure the min max of width and height are not larger
            # or smaller than the image boundary
            output[:, [1, 3]] -= (params["height"] - scaling_factor *
                                  im_dim_list[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (params["height"] - scaling_factor *
                                  im_dim_list[:, 1].view(-1, 1))/2
            output[:, 1:5] /= scaling_factor
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]],
                                                0.0, im_dim_list[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]],
                                                0.0, im_dim_list[i, 1])

            list(map(lambda x: detection_write(x, original_imgs,
                                               params["classes"]),
                     output))
        det_names = pd.Series(imlist).apply(
                           lambda x: "{}/{}".format(det, x.split("/")[-1]))
        list(map(cv2.imwrite, det_names, original_imgs))
        if model.params['cuda']:
            torch.cuda.empty_cache()


def simgle_img_map(model, output, labels):
    actual_num_labels = 0
    all_detections = [[[np.array([]) for _ in range(
            model.params['num_classes'])]]]

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
        for c in range(model.params['num_classes']):
            all_detections[0][-1][c] = pred_boxes[
                                              pred_labels == c]
    all_annotations = []
    for label_ in labels:
        all_annotations.append(
                            [np.array([]) for _ in range(
                                    model.params['num_classes'])]
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
            annotation_boxes *= model.params['height']

            for label in range(model.params['num_classes']):
                all_annotations[-1][label] =\
                    annotation_boxes[annotation_labels == label, :]
    # if train it results consists mAP, average_precisions map_frame
    # else: mAP, average_precisions
    actual_num_labels = np.max([actual_num_labels, num_labels])
    print(f"actual_num_labels : {actual_num_labels}")
    mAP, ap = compute_map(all_detections, all_annotations, conf_index=0,
                          map_frame=None, train=True,
                          actual_num_labels=actual_num_labels,
                          params=model.params)
    return mAP, ap


def get_single_img_map(cfg_path, det, label_csv_mame, params):
    map_frame = pd.DataFrame(columns=['mAP', 'ap'])
    assert params["height"] % 32 == 0
    assert params["height"] > 32
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(params, blocks)
    # model.load_weights("../4Weights/yolov3.weights")
    if params['cuda']:
        model = model.cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    # yolo v3 down size the imput images 32 strides, therefore the input needs
    # to be a multiplier of 32 and > 32

    # put to evaluation mode so that gradient wont be calculated
    model.eval()
    transform = transforms.Compose(
            [LetterBoxImage_cv([params['height'], params['height']]),
             ImgToTensorCv(), transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])

    data = CustDataCV(label_csv_mame, transform=transform,
                      detection_phase=True)
    data_loader = DataLoader(data, shuffle=False,
                             batch_size=1,
                             collate_fn=my_collate_detection,
                             num_workers=0, worker_init_fn=worker_init_fn)
    for original_imgs, images, imlist, im_dim_list, labels in data_loader:
        if model.params['cuda']:
            images = images.cuda()
        prediction = model(images)
        output = filter_results(prediction, params)
        mAP, ap = simgle_img_map(model, output, labels)
        map_frame.loc[imlist[-1], 'mAP'] = mAP
        map_frame.loc[imlist[-1], 'ap'] = ap
    return map_frame


def main(weight_path):
    # change cuda to True if you wish to use gpu
    det = "../2ProcessedData/"
    cfg_path = "../4Others/yolo.cfg"
    label_csv_mame = '../1TestData/label.csv'
    name_list = ["img_name", "c", "gx", "gy", "gw", "gh"]
    test_img_txt_path = "../1TestData/*.txt"
    prep_labels(test_img_txt_path, name_list, label_csv_mame)
    params = prep_params(weight_path, label_csv_mame)
    detection(cfg_path, det, label_csv_mame, params)
    df = get_single_img_map(cfg_path, det, label_csv_mame, params)
    return df


if __name__ == '__main__':
    weight_path = "../4TrainingWeights/human_eye/tune/2019-02-03_01_02_32.525378/2019-02-03_02_47_27.382756.txt"
    df = main(weight_path)
