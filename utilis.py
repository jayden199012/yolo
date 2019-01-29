import os
import torch
import torch.nn as nn
import numpy as np
import datetime
import json
import cv2
import glob
import pandas as pd
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from generate_anchors import generate_anchor
import logging

# add this two libraries if you are not using windpws and wish to use mp
# from itertools import repeat
# from torch.multiprocessing import Pool


class empty_layer(nn.Module):
    def __init__(self):
        super().__init__()


class yolo_layer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


def load_classes(namesfile):
    with open(namesfile, "r") as file:
        names = file.read().split("\n")[:-1]
    return names


# pass config file into a list
def parse_cfg(cfgfile):
    with open(cfgfile, "r") as file:
        lines = file.read().split('\n')
        # get rid of lines that are blanks for comments
        lines = [line for line in lines if (len(line) > 0) and
                 (line[0] != '#')]

        # get ride of white spaces from the back and front
        lines = [line.lstrip().rstrip() for line in lines]
        temp_block = {}
        blocks = []

        # loop through the lines to fill up the blocks list
        for line in lines:
            if line[0] == "[":
                if len(temp_block) != 0:
                    blocks.append(temp_block)
                    temp_block = {}
                temp_block['type'] = line[1:-1].lstrip().rstrip()
            else:
                key, value = line.split('=')

                # set key value to the dictionary, remove possible spaces
                # near '='
                temp_block[key.rstrip()] = value.lstrip()

        # append the final block when the loop is over
        blocks.append(temp_block)
#        blocks[0]["steps"] = blocks[0]["steps"].split(",")
#        blocks[0]["steps"] = [int(x) for x in blocks[0]["steps"]]
#        blocks[0]["scales"] = blocks[0]["scales"].split(",")
#        blocks[0]["scales"] = [float(x) for x in blocks[0]["scales"]]
#        int_rows = ["batch", "max_batches", "height", "width", "channels",
#                    "epochs", "burn_in", "max_batches"]
#        float_rows = ["momentum", "decay", "saturation", "exposure", "hue",
#                      "learning_rate", "rand_crop"]
#        for rows in int_rows:
#            blocks[0][rows] = int(blocks[0][rows])
#        for rows in float_rows:
#            blocks[0][rows] = float(blocks[0][rows])
        return blocks


def conv_layer_handling(module, index, layer, in_channel, layer_type_dic):
    activation = layer["activation"]
    try:
        batch_norm = layer["batch_normalize"]
        bias = False  # bias term is already included in batch_norm
                      # gamma * normalized(x) + bias
    except KeyError as e:
        batch_norm = 0
        bias = True

    out_channels = int(layer["filters"])
    kernel_size = int(layer["size"])
    padding = 0 if kernel_size == 1 else 1  # padding not require for 1*1 conv
    stride = int(layer["stride"])
    conv = nn.Conv2d(in_channel, out_channels, kernel_size, stride, padding,
                     bias=bias)
    module.add_module("conv_{}".format(index), conv)

    # batch norm layer if applicable
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        module.add_module("bach_norm_{}".format(index), bn)

    if activation == "leaky":
        activn = nn.LeakyReLU(0.1)
        module.add_module("leaky_{}".format(index), activn)
    layer_type_dic["conv"].append(index)
    return out_channels


def upsample_layer_handling(module, index, layer, layer_type_dic):
    stride = int(layer["stride"])
    upsample = nn.Upsample(scale_factor=stride, mode="nearest")
    module.add_module("upsample_{}".format(index), upsample)
    layer_type_dic['upsampling'].append(index)


def route_layer_handling(module, index, layer, out_channels_list,
                         layer_type_dic):
    items = [int(item) for item in (layer["layers"].split(','))]
    first_layer = index + items[0]
    out_channel = out_channels_list[first_layer]
    if len(items) == 1:
        layer_type_dic['referred_relationship'][index] = (first_layer)
        layer_type_dic['route_1'].append(index)
    elif len(items) == 2:
        out_channel += out_channels_list[items[1]]  # sum of c from two layers
        layer_type_dic['referred'].append(items[1])
        layer_type_dic['referred_relationship'][index] = \
            ((first_layer, items[1]))   # [items[1] is the second layer
        layer_type_dic['route_2'].append(index)
    else:
        raise Exception('Route layer is not behaving as we planned, please \
                        change the code according.')
    route = empty_layer()
    module.add_module("route_{}".format(index), route)
    layer_type_dic['referred'].append(index + items[0])
    return out_channel


def shortcut_layer_handling(module, index, layer, layer_type_dic):
    short_cut = empty_layer()
    from_layer = int(layer["from"])
    module.add_module("short_cut_{}".format(index), short_cut)
    layer_type_dic['shortcut'].append(index)
    # add the layer that is referred in the short cut layer
    layer_type_dic['referred'].append(index + from_layer)
    layer_type_dic['referred_relationship'][index] = (index + from_layer)


def yolo_layer_handling(params, module, index, layer, layer_type_dic):
    a = len(layer_type_dic['yolo']) * params['num_anchors']
    layer["mask"] = range(a, a + params['num_anchors'])
    anchors = params['anchors'][layer["mask"]]
    yolo = yolo_layer(anchors)
    module.add_module("yolo_{}".format(index), yolo)
    layer_type_dic['yolo'].append(index)


def create_module(params, blocks):
    module_list = nn.ModuleList()
    layer_type_dic = {"conv": [],
                      "upsampling": [],
                      "shortcut": [],
                      "route_1": [],
                      "route_2": [],
                      "yolo": [],
                      "referred": [],
                      "referred_relationship": {}}
    params['width'] = params['height']
    in_channel = params['channels']
    out_channels_list = []

    for index, layer in enumerate(blocks):
        module = nn.Sequential()

        # conv layer
        if layer["type"] == "convolutional":
            in_channel = conv_layer_handling(module, index, layer, in_channel,
                                             layer_type_dic)

        # up sampling layer
        elif layer["type"] == "upsample":
            upsample_layer_handling(module, index, layer, layer_type_dic)

        # route layer
        elif layer["type"] == "route":
            in_channel = route_layer_handling(module, index, layer,
                                              out_channels_list,
                                              layer_type_dic)

        # shor cut layer
        elif layer["type"] == "shortcut":
            shortcut_layer_handling(module, index, layer, layer_type_dic)

        # yolo layer
        elif layer["type"] == "yolo":
            yolo_layer_handling(params, module, index, layer, layer_type_dic)

        out_channels_list.append(in_channel)
        module_list.append(module)

    for index, value in layer_type_dic.items():
        if index != "referred_relationship":
            layer_type_dic[index] = list(layer_type_dic[index])

    return layer_type_dic, module_list


def box_iou(box1, box2):
    ''' Both boxes need to be a 2d tensor '''
    b1x_min, b1y_min, b1x_max, b1y_max = box1[:, 0], box1[:, 1],\
        box1[:, 2], box1[:, 3]
    b2x_min, b2y_min, b2x_max, b2y_max = box2[:, 0], box2[:, 1], \
        box2[:, 2], box2[:, 3]

    # find the co-ordinates of the intersection rectangle
    inter_box_xmin = torch.max(b1x_min, b2x_min)
    inter_box_ymin = torch.max(b1y_min, b2y_min)
    inter_box_xmax = torch.min(b1x_max, b2x_max)
    inter_box_ymax = torch.min(b1y_max, b2y_max)

    # intersection area
    inter_area = torch.clamp((inter_box_xmax - inter_box_xmin + 1),
                             min=0) * \
        torch.clamp((inter_box_ymax - inter_box_ymin + 1), min=0)

    # intersection area
    box1_area = (b1x_max - b1x_min + 1) * (b1y_max - b1y_min + 1)
    box2_area = (b2x_max - b2x_min + 1) * (b2y_max - b2y_min + 1)
    union_area = box1_area + box2_area - inter_area

    # iou
    iou = inter_area / union_area
    return iou


def np_box_iou(box1, box2):
    ''' Both boxes need to be a 2d array '''
    b1x_min, b1y_min, b1x_max, b1y_max = box1[:, 0], box1[:, 1],\
        box1[:, 2], box1[:, 3]
    b2x_min, b2y_min, b2x_max, b2y_max = box2[:, 0], box2[:, 1], \
        box2[:, 2], box2[:, 3]

    # find the co-ordinates of the intersection rectangle
    inter_box_xmin = np.maximum(b1x_min, b2x_min)
    inter_box_ymin = np.maximum(b1y_min, b2y_min)
    inter_box_xmax = np.minimum(b1x_max, b2x_max)
    inter_box_ymax = np.minimum(b1y_max, b2y_max)

    # intersection area
    inter_area = (np.clip((inter_box_xmax - inter_box_xmin + 1), a_max=None,
                          a_min=0) *
                  np.clip((inter_box_ymax - inter_box_ymin + 1), a_max=None,
                          a_min=0))

    # intersection area
    box1_area = (b1x_max - b1x_min + 1) * (b1y_max - b1y_min + 1)
    box2_area = (b2x_max - b2x_min + 1) * (b2y_max - b2y_min + 1)
    union_area = box1_area + box2_area - inter_area

    # iou
    iou = inter_area / union_area
    return iou


def filter_img_class(cls, image_pred, ind, nms_thesh):
    image_pred_class = image_pred[image_pred[:, 6] == cls]

    # sort the detections such that the entry with the maximum
    # objectness confidence is at the top
    conf_sort_index = torch.sort(image_pred_class[:, 4],
                                 descending=True)[1]
    image_pred_class = image_pred_class[conf_sort_index]

    # Number of detections
    idx = image_pred_class.size(0)

    for i in range(idx):
        # Get the IOUs of all boxes that come after the one we are
        # looking at in the loop
        try:
            ious = box_iou(image_pred_class[i, :4].unsqueeze(0),
                           image_pred_class[i+1:, :4])

        except (IndexError, ValueError):
            break

        # true will return 1 and false will return 0
        # Zero out all the detections that have IoU > treshhold
        iou_mask = (ious < nms_thesh).float().unsqueeze(1)
        # only remove rows after the current index
        image_pred_class[i+1:] *= iou_mask

        # Remove the non-zero entries
        non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
        image_pred_class = image_pred_class[non_zero_ind].view(-1, 8)

    # Repeat the batch_id for the same class in the image
    batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
    seq = batch_ind, image_pred_class
    return seq


def filter_results(prediction, params):
    conf_mask = (prediction[:, :, 4] >
                 params['confidence']).float().unsqueeze(2)
    prediction = prediction*conf_mask
    y_coord = prediction[:, :, 1].unsqueeze(2)

    # create a new tensor that has the identical dtype as prediction tensor
    box_corners = prediction.new(prediction.shape)

    # convert tensor format from center x, center y to xmin,ymin
    box_corners[:, :, :2] = prediction[:, :, :2] - prediction[:, :, 2:4]/2

    # convert tensor format from width. height xmax,ymax
    box_corners[:, :, 2:4] = prediction[:, :, :2] + prediction[:, :, 2:4]/2

    # copy the tr`sfored numbers to prediction
    prediction[:, :, :4] = box_corners[:, :, :4]

    prediction = torch.cat((prediction, y_coord), 2)

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        # to get rid of the rows that below the confidence threshold
        image_pred = prediction[ind]
        image_pred = image_pred[image_pred[:, 4] > 0]
        # in case there is no predictions above our confidence level
        if image_pred.shape[0] == 0:
            continue
        # torch.max(imput, dim) which returns the number and the index
        max_conf, class_index = torch.max(
                image_pred[:, 5:5 + params['num_classes']], 1)

        # N * 1 tensor
        max_conf = max_conf.float().unsqueeze(1)
        class_index = class_index.float().unsqueeze(1)

        # concat xmin,ymin,xmax,ymax,object conf together with
        # class prob, class index,  y_centre
        seq = (image_pred[:, :5], max_conf, class_index,
               image_pred[:, -1].unsqueeze(1))
        image_pred = torch.cat(seq, 1)

        # find the unique index in the picture thus find the classes
        img_classes = torch.unique(image_pred[:, 6])

        # replace for cls loop if you are not using windows,
        # for multiprocessing

#        with Pool(4) as pool:
#            output = torch.cat(
#                    pool.starmap(filter_img_class, zip(
#                            img_classes, repeat(image_pred), repeat(ind),
#                            repeat(nms_conf))), 1)

        for cls in img_classes:
            # erform NMS
            # seq is a tuple of  batch_ind, min,ymin,xmax,ymax,object conf,
            # class prob, class index,  y_centre
            seq = filter_img_class(cls, image_pred, ind, params['nms_thesh'])
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except NameError:
        return 0


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def letterbox_image(img, inp_dim, inp_channel=3):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], inp_channel), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,
           (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image

    return canvas


class LetterboxImage_cv():
    def __init__(self, inp_dim, inp_channel=3):
        self.w, self.h = inp_dim
        self.inp_channel = inp_channel

    def __call__(self, img):
        img_w, img_h = img.shape[1], img.shape[0]
        new_w = int(img_w * min(self.w/img_w, self.h/img_h))
        new_h = int(img_h * min(self.w/img_w, self.h/img_h))
        resized_image = cv2.resize(img, (new_w, new_h),
                                   interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.h, self.w, self.inp_channel), 128)
        canvas[(self.h-new_h)//2:(self.h-new_h)//2 + new_h,
               (self.w-new_w)//2:(self.w-new_w)//2 + new_w, :] = resized_image

        # normalize image
        canvas = (canvas - canvas.min())/(canvas.max()-canvas.min())
        return canvas


class LetterboxImage():
    def __init__(self, inp_dim):
        self.w, self.h = inp_dim

    def __call__(self, img):
        img = np.array(img)
        img_w, img_h = img.shape[1], img.shape[0]
        new_w = int(img_w * min(self.w/img_w, self.h/img_h))
        new_h = int(img_h * min(self.w/img_w, self.h/img_h))
        resized_image = cv2.resize(img, (new_w, new_h),
                                   interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.h, self.w, 3), 128)
        canvas[(self.h-new_h)//2:(self.h-new_h)//2 + new_h,
               (self.w-new_w)//2:(self.w-new_w)//2 + new_w, :] = resized_image
        return canvas


class ImgToTensorCv():
    def __call__(self, img):
        # change from h w c to c h w for pytorch imput
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img = torch.FloatTensor(img)
        return img


class ImgToTensor():
    def __call__(self, img):
        # change from w h c to c h w for pytorch imput
        img = img[:, :, ::-1].transpose((2, 1, 0)).copy()
        img = torch.FloatTensor(img)
        return img


def prep_image(img, inp_dim):
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def prep_labels(img_txt_path, name_list, label_csv_mame, selected_cls=False):
    out_file = pd.DataFrame(columns=name_list)
    read_files = glob.glob(img_txt_path)
    index = 0
    for file in read_files:
        with open(file, "r") as infile:
            for line in infile:
                items = line.split()
                for i, col in enumerate(name_list):
                    if i == 0:
                        out_file.loc[index, col] = file.replace('txt', 'jpg')
                    else:
                        out_file.loc[index, col] = items[i-1]
                index += 1
    if selected_cls:
        out_file = out_file[out_file.c.isin(selected_cls)]
    out_file.to_csv(label_csv_mame, index=False)


# to combine label and images together
def my_collate(batch):
    image = torch.stack([item["image"] for item in batch], 0)
    labels = [item["label"] for item in batch]
    samples = {'image': image, 'label': labels}
    return samples


# to combine label and images together during detection phase
def my_collate_detection(batch):
    original_images = [item["ori_image"] for item in batch]
    images = torch.stack([item["image"] for item in batch], 0)
    img_names = [item["img_name"] for item in batch]
    dims = [item["dim"] for item in batch]
    labels = [item["labels"] for item in batch]
    return original_images, images, img_names, dims, labels


def draw_boxes(image, boxes):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * image.shape[1]
    boxes[:, [2, 4]] = boxes[:, [2, 4]] * image.shape[0]
    colors = {i: np.random.rand(3,) for i in range(len(boxes))}
    for i, box in enumerate(boxes):
        x, y, w, h = (box[1] - box[3]/2), (box[2] - box[4]/2), box[3], box[4]
        xmin, ymin, xmax, ymax = x, y, x+w, y+h
        p = Polygon(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)),
                    fc=(colors[i][0], colors[i][1], colors[i][2], 0.35),
                    ec=(colors[i][0], colors[i][1], colors[i][2], 0.95), lw=3)
        ax.add_patch(p)
        ax.axis('off')
    ax.imshow(image)


def write(x, results, classes):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[7])
#    color = random.choice(colors)
    color = 2
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def detection_write(x, results, classes):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    img = results[int(x[0])]
    cls = int(x[7])
#    color = random.choice(colors)
    color = 2
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def move_images(label_name, to_path, action_fn, action="copy", **kwargs):
    labels_df = action_fn(label_name, **kwargs)
    if action == "move":
        for images in labels_df.iloc[:, 0]:
            shutil.move(images, to_path)
            shutil.move(images[:-3] + 'txt', to_path)
    elif action == "copy":
        for images in labels_df.iloc[:, 0]:
            shutil.copy(images, to_path)
            shutil.copy(images[:-3] + 'txt', to_path)


def move_images_cv(dfs, to_paths, name_list):
    for df, to_path in zip(dfs, to_paths):
        if not os.path.exists(to_path):
                os.makedirs(to_path)
        for images in df.iloc[:, 0]:
            shutil.copy(images, to_path)
            shutil.copy(images[:-3] + 'txt', to_path)
        prep_labels(f"{to_path}*.txt", name_list,
                    f"{to_path}label.csv")


def resize_all_img(new_w, new_h, from_path, to_path):
    '''
    This function resize images into new sizes and make a copy of them to a
    new path
    '''
    images = glob.glob(from_path)
    for img in images:
        image = Image.open(img)
        resized_image = image.resize((new_w, new_h), Image.ANTIALIAS)
        img_name = img.split('\\')[-1]
        save_path = to_path + img_name
        if not os.path.isfile(save_path):
            resized_image.save(save_path)
        else:
            raise NameError(f"{save_path} already exists")


def worker_init_fn(worker_id):
    np.random.seed(worker_id)


def _save_checkpoint(model, save_txt=True):
    # global best_eval_result
    time_now = str(datetime.datetime.now()).replace(
                                   " ",  "_").replace(":",  "_")
    checkpoint_path = os.path.join(model.params["sub_working_dir"],
                                   time_now + ".pth")
    model.params["pretrain_snapshot"] = checkpoint_path
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model checkpoint saved to {checkpoint_path}")
    if save_txt:
        with open(f'{model.params["sub_working_dir"]}/{time_now}.txt', "w"
                  ) as file:
            params_ = model.params.copy()
            params_['anchors'] = params_['anchors'].tolist()
            file.write(json.dumps(params_, indent=4))
            del params_


def prep_params(params_dir, label_csv_mame, experiment_params=False):
    with open(params_dir) as fp:
        params = json.load(fp)
    if experiment_params:
        params = {**params, **experiment_params}
    params['width'] = params['height']
    if type(params['anchors']) == list:
        params['anchors'] = np.array(params['anchors'])
    else:
        params['anchors'] = generate_anchor(label_csv_mame,
                                            params['width'],
                                            params['height'],
                                            num_clusters=params[
                                                    'num_anchors']*3)
    params['classes'] = load_classes('../4Others/color_ball.names')
    params['conf_list'] = list(np.arange(start=0.2, stop=0.95, step=0.025))
    return params





def prep_txt(txt_dir, img_dir, labels_csv, label_name):
    labels = pd.read_csv(labels_csv)
    labels = labels.iloc[:, ]
    imgs = glob.glob(f"{img_dir}*jpg")
    imgs_s = pd.Series(imgs)
    imgs_s = imgs_s.apply(lambda x: x.split('\\')[1].replace('.jpg', ''))
    labels = labels.iloc[((labels.ImageID.isin(imgs_s)) &
                          (labels.LabelName == label_name)).values,
                         [0, 2, 4, 5, 6, 7]]
    labels.LabelName = 0
    w = abs(labels.XMax - labels.XMin)
    h = abs(labels.YMax - labels.YMin)
    x = labels.XMin + w/2
    y = labels.YMin + h/2
    labels = pd.concat([labels, x.rename('x'), y.rename('y'), w.rename('w'),
                        h.rename('h')], axis=1)
    labels = labels.drop(columns=['XMin', 'XMax', 'YMin', 'YMax'], axis=1)
    for img in imgs_s:
        labels.loc[labels.ImageID == img, labels.columns != 'ImageID'].to_csv(
                f'{img_dir}{img}.txt', header=None, index=None, sep=' ',
                mode='w')


#labels_csv = '../4Others/OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv'
#img_dir = '../train/eye/'
#label_name = '/m/014sv8'
#image = np.array(Image.open('../train/eye/000cf4b56061f60f.jpg'))
#boxes = pd.read_csv('../train/eye/000cf4b56061f60f.txt', sep=" ", header=None).values
#draw_boxes(image, boxes)
#        
#x = cv2.imread('../1TrainData/4aeeac7ac9fed526.jpg')
#x = Image.open('../1TrainData/4aeeac7ac9fed526.jpg')
#x = np.array(Image.open('../1TrainData/4aeeac7ac9fed526.jpg'))
#len(x.shape)
#
#x = color.gray2rgb(x)
#Image.Image.convert("RGB", x)
