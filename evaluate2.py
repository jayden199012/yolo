from __future__ import division

import torch
import time
import pandas as pd
import numpy as np
from utilis import compute_ap, filter_results, load_classes
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import parse_cfg,  my_collate
from yolo_v3 import yolo_v3
from data import CustData


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua



def eval_score(model, dataloader, cuda):
    model.eval()
    with torch.no_grad():
        for step, samples in enumerate(dataloader):
            if cuda:
                images, labels = samples["image"].to('cuda'), samples["label"]
            else:
                images, labels = samples["image"], samples["label"]
            losses = model(images, cuda, is_training=True, labels=labels)
            if not step:
                evaluate_loss = losses.copy()
            else:
                evaluate_loss = [a + b for a, b in zip(evaluate_loss,
                                                       losses)]

            model.train()
            return evaluate_loss


def compute_map(num_classes, classes, all_detections, all_annotations,
                conf_index, confidence, nms_conf, map_frame, train):
    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0
    
        for i in range(len(all_annotations)):
            detections = all_detections[conf_index][i][label]
            annotations = all_annotations[i][label]
    
            num_annotations += annotations.shape[0]
            detected_annotations = []
    
            for *bbox, score in detections:
                scores.append(score)
                
    
                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

#                overlaps = np_box_iou(
#                        np.expand_dims(bbox, axis=0), annotations)
                overlaps = bbox_iou_numpy(
                        np.expand_dims(bbox, axis=0), annotations)
                
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]
    
                if max_overlap >= nms_conf and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)
    
        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue
    
        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
    
        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)
    
        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    
        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
    mAP = np.mean(list(average_precisions.values()))
    if not train:
        print(f"Average Precisions when confidence = {confidence}:")
        for c, ap in zip(classes, average_precisions.values()):
            map_frame.loc[c, confidence] = ap
            print(f"+ Class '{c}' - AP: {ap}")
        map_frame.loc["mAP", confidence] = mAP
        print(f"mAP: {mAP}")
        print(f"average_precisions: {average_precisions}")
        return mAP, average_precisions.values(), map_frame
    else:
        return mAP, average_precisions.values()


def get_map(model, dataloader, cuda, conf_list, nms_conf, classes,
            train=False):
    if not train:
        rows = classes + ["mAP"]
        map_frame = pd.DataFrame(index=rows, columns=conf_list)
    else:
        map_frame = None
    model.eval()
    num_classes = model.num_classes
    all_detections = []
    len_conf_list = len(conf_list)
    for _ in range(len_conf_list):
        all_detections.append([])
    all_annotations = []
    sum_map = 0
    with torch.no_grad():
        for samples in dataloader:
            if cuda:
                image, labels = samples["image"].to('cuda'), samples["label"]
            else:
                image, labels = samples["image"], samples["label"]

            img_size = image.shape[-1]
            outputs = model(image, cuda)

            for conf_index, confidence in enumerate(conf_list): 
                for img in outputs:
                    all_detections[conf_index].append(
                        [np.array([]) for _ in range(num_classes)]
                        )                  
                    outputs_ = filter_results(img.unsqueeze(0), confidence,
                                              num_classes, nms_conf)
    
                    if outputs_ is not 0:
                        # Get predicted boxes, confidence scores and labels
                        pred_boxes = outputs_[:, 1:6].cpu().numpy()
                        scores = outputs_[:, 5].cpu().numpy()
                        pred_labels = outputs_[:, 7].cpu().numpy()
    
                        # Order by confidence
                        sort_i = np.argsort(scores)
                        pred_labels = pred_labels[sort_i]
                        pred_boxes = pred_boxes[sort_i]
                        for c in range(num_classes):
                            all_detections[conf_index][-1][c] = pred_boxes[
                                                                pred_labels == c]

            # get all labels for a batch

            for label_ in labels:
                all_annotations.append(
                                    [np.array([]) for _ in range(num_classes)]
                                        )

                if any(label_[:, -1] > 0):

                    annotation_labels = label_[label_[:, -1] > 0, 0]
                    _annotation_boxes = label_[label_[:, -1] > 0, 1:]
    
                    # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                    annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                    annotation_boxes *= img_size
        
                    for label in range(num_classes):
                        all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]
        for conf_index, confidence in enumerate(conf_list):
            results = compute_map(num_classes, classes, all_detections,
                                  all_annotations, conf_index, confidence,
                                  nms_conf, map_frame, train)
            if conf_index == 0:
                sum_ap = results[1]
            else:
                sum_ap = [a + b for a, b in zip(sum_ap, results[1])]
            sum_map += results[0]
    model.train()
    ap = [x / len_conf_list for x in sum_ap]
    mAP = sum_map/len_conf_list
    if train:
        return sum_ap, sum_map, ap, mAP
    else:
        return sum_ap, sum_map, ap, mAP, map_frame




def main():
    cuda = True
    cfg_path = "../4Others/color_ball.cfg"
    test_root_dir = "../1TestData"
    test_label_csv_mame = '../1TestData/label.csv'
    classes = load_classes('../4Others/color_ball.names')
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    conf_list = np.arange(start=0.2, stop=0.95, step=0.025)
    checkpoint_path = "../4TrainingWeights/2018-11-07_23_13_38.391465/2018-11-08_02_45_20.195250_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    test_transform = transforms.Compose(
            [transforms.Resize(model.net["height"]), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    test_data = CustData(test_label_csv_mame, test_root_dir,
                         transform=test_transform)

    test_loader = DataLoader(test_data, shuffle=True,
                             batch_size=1,
                             collate_fn=my_collate,
                             num_workers=4)
    start = time.time()
    sum_ap, sum_map, ap, mAP, map_frame = get_map(model, test_loader, cuda,
                                                  conf_list, 0.5, classes,
                                                  train=False)
    print(time.time() - start)
    return sum_ap, sum_map, ap, mAP, map_frame


if __name__ == "__main__":
    maP, df = main()
