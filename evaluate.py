from __future__ import division

import torch
import time
import pandas as pd
import numpy as np
from utilis import compute_ap, filter_results
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import parse_cfg,  my_collate,  prep_params
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

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1),
                    box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1),
                    box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]),
                        axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def eval_score(model, dataloader):
    '''
    This function is to get average test set losss
    Argument:
        model
        dataLoaderL data lodaer objects
    '''
    model.eval()
    with torch.no_grad():
        for step, samples in enumerate(dataloader):
            if model.params['cuda']:
                images, labels = samples["image"].to('cuda'), samples["label"]
            else:
                images, labels = samples["image"], samples["label"]
            losses = model(images, is_training=True, labels=labels)
            if not step:
                evaluate_loss = losses.copy()
            else:
                evaluate_loss = [a + b for a, b in zip(evaluate_loss,
                                                       losses)]

            model.train()
            return evaluate_loss


def compute_map(all_detections, all_annotations, conf_index,
                map_frame, train, actual_num_labels, params):
    average_precisions = {}
    for label in range(params['num_classes']):
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

                if max_overlap >= params['iou_conf'] and \
                        assigned_annotation not in detected_annotations:
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
        precision = true_positives / np.maximum(true_positives +
                                                false_positives,
                                                np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

#    mAP = np.mean(list(average_precisions.values()))
    mAP = sum(list(average_precisions.values()))/actual_num_labels
    if not train:
        print(f"Average Precisions when confidence = {params['confidence']}:")
        for c, ap in zip(params['classes'], average_precisions.values()):
            map_frame.loc[c, params['confidence']] = ap
            print(f"+ Class '{c}' - AP: {ap}")
        print(f"mAP: {mAP}")
        map_frame.loc['mAP', params['confidence']] = mAP
        print(f"average_precisions: {average_precisions}")
        return mAP, list(average_precisions.values()), map_frame
    else:
        return mAP, list(average_precisions.values())


def get_map(model, dataloader, train=False, loop_conf=False, confidence=False):
    actual_num_labels = 0
    if confidence:
        loop_conf = confidence
    elif loop_conf:
        loop_conf = model.params['conf_list']
    else:
        loop_conf = [model.params['specific_conf']]
    if not train:
        rows = model.params['classes'] + ["mAP"]
        map_frame = pd.DataFrame(index=rows, columns=loop_conf)
    else:
        map_frame = None
    model.eval()
    num_classes = model.params['num_classes']
    all_detections = []
    specific_conf_map = None
    specific_conf_ap = None
    len_conf_list = len(loop_conf)
    for _ in range(len_conf_list):
        all_detections.append([])
    all_annotations = []
    with torch.no_grad():
        for samples in dataloader:
            if model.params['cuda']:
                image, labels = samples["image"].to('cuda'), samples["label"]
            else:
                image, labels = samples["image"], samples["label"]

            img_size = image.shape[-1]
            outputs = model(image)
            for conf_index, confidence in enumerate(loop_conf):
                # gets output at each object confidence
                model.params['confidence'] = confidence
                for img in outputs:
                    all_detections[conf_index].append(
                        [np.array([]) for _ in range(num_classes)]
                        )
                    outputs_ = filter_results(img.unsqueeze(0), model.params)

                    # n our model if no results it outputs int 0
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
                    num_labels = len(np.unique(annotation_labels))
                    # Reformat to x1, y1, x2, y2 and rescale to image dim
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                    annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                    annotation_boxes *= img_size

                    for label in range(num_classes):
                        all_annotations[-1][label] = annotation_boxes[
                                annotation_labels == label, :]
                actual_num_labels = np.max([actual_num_labels, num_labels])
        for conf_index, confidence in enumerate(loop_conf):
            model.params['confidence'] = confidence
            print(f"Running for object confidence : {confidence}")
            # if train it results consists mAP, average_precisions map_frame
            # else: mAP, average_precisions
            print(f"actual_num_labels : {actual_num_labels}")
            results = compute_map(all_detections, all_annotations, conf_index,
                                  map_frame, train, actual_num_labels,
                                  model.params)
            if conf_index == 0:
                best_map = results[0]
                best_ap = results[1]
                best_conf = confidence

            else:
                if results[0] > best_map:
                    best_map = results[0]
                    best_ap = results[1]
                    best_conf = confidence
            if np.round(confidence, 3) == model.params['specific_conf']:
                specific_conf_map = results[0]
                specific_conf_ap = results[1]

    model.train()

    if train:
        return best_map, best_ap, best_conf, specific_conf_map,\
               specific_conf_ap
    else:
        return best_map, best_ap, best_conf, specific_conf_map,\
               specific_conf_ap, map_frame


# %%

def main(param_dir):
    cfg_path = "../4Others/yolo.cfg"
    test_root_dir = "../2CvTrain"
    test_label_csv_mame = '../2CvTrain/label.csv'
    blocks = parse_cfg(cfg_path)
    label_csv_mame = '../2CvTrain/label.csv'
    params = prep_params(param_dir, label_csv_mame)
    from yolo_v3 import yolo_v3
    model = yolo_v3(params, blocks)
    test_transform = transforms.Compose(
            [transforms.Resize(model.params["height"]), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    test_data = CustData(test_label_csv_mame,
                         transform=test_transform)

    test_loader = DataLoader(test_data, shuffle=False,
                             batch_size=model.params["batch_size"],
                             collate_fn=my_collate,
                             num_workers=0)
    start = time.time()
    best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
        map_frame = get_map(model, test_loader, train=False, loop_conf=False)
    print(time.time() - start)
    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame


if __name__ == "__main__":
    start = time.time()
    param_dir = '../4Others/params.txt'
    best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame = main(param_dir)
    print(f"time spend: {time.time()-start}")
    print(f"Best mAP is : {best_map}")
    print(f"Best AP is : {best_ap}")
    print(f"specific conf mAP is : {specific_conf_map}")
    print(f"specific conf AP is : {specific_conf_ap}")
    print(f"Best conf is : {best_conf}")

