import numpy as np
import math
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utilis import letterbox_image, draw_boxes
from skimage import color
import cv2


class CustData(Dataset):

    def __init__(self, csv_file,
                 pre_trans=None, transform=None, post_trans=None,
                 detection_phase=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            pre_trans (callable, optional): Custom transforms that applies
            to both the images and the labels
            transform (callable, optional): Optional transform to be applied
            on a sample.
            post_trans (callable, optional): Custom transforms that applies
            to both the images and the labels
            The reason why I break transforms into 3 part is because the
            in build torch vision transforms can only do transfromations that
            will not affect the labels. Thus if you wish to do custom
            transformations before or after torch vision transforms, please
            specify seperately
        """
        self.label_frame = pd.read_csv(csv_file)
        self.pre_trans = pre_trans
        self.transform = transform
        self.post_trans = post_trans
        self.list_IDs = self.label_frame.iloc[:, 0].unique()
        self.detection_phase = detection_phase

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        img_name = self.list_IDs[idx]
        image = Image.open(img_name)
        if self.detection_phase:
            ori_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            w, h = image.size[:2]
        labels = self.label_frame.iloc[:, 1:][self.label_frame.iloc[
                :, 0] == img_name].astype('float').values.reshape(-1, 5)
        if self.pre_trans:
            image, labels = self.pre_trans(image, labels)
        if self.transform:
            image = self.transform(image)

        if self.post_trans:
            image, labels = self.post_trans(image, labels)
        if self.detection_phase:
            samples = {'ori_image': ori_image, 'image': image,
                       'img_name': img_name.split('\\')[-1], 'dim': (w, h),
                       'labels': labels}
        else:
            samples = {'image': image, 'label': labels}
        return samples


class CustDataCV(Dataset):

    def __init__(self, csv_file, pre_trans=None, transform=None, detection_phase=False):
        """
           Same as CustData but opens image using open cv instead of pillow
        """
        self.label_frame = pd.read_csv(csv_file)
        self.pre_trans = pre_trans
        self.transform = transform
        self.list_IDs = self.label_frame.iloc[:, 0].unique()
        self.detection_phase = detection_phase

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        img_name = self.list_IDs[idx]
        image = cv2.imread(img_name)
        if self.detection_phase:
            ori_image = image.copy()
            h, w = image.shape[:2]
        labels = self.label_frame.iloc[:, 1:][self.label_frame.iloc[
                :, 0] == img_name].astype('float').values.reshape(-1, 5)
        if self.pre_trans:
            image, labels = self.pre_trans(image, labels)
        if self.transform:
            image = self.transform(image)
        if self.detection_phase:
            samples = {'ori_image': ori_image, 'image': image,
                       'img_name': img_name.split('\\')[-1], 'dim': (w, h),
                       'labels': labels}
        else:
            samples = {'image': image, 'label': labels}
        return samples


class RandomCrop:

    def __init__(self, jitter=0.2, inp_dim=416):
        self.jitter = jitter
        self.inp_dim = inp_dim

    def __call__(self, image, labels):
        # read imgage using PIL formate (w, h, c)
#        labels = np.array([[0.,        0.350629,  0.20875,   0.034592,  0.015    ],
#             [0.,        0.4067085, 0.2103125, 0.031447,  0.015625 ],
#             [0. ,       0.610063,  0.1771875, 0.037736,  0.016875 ],
#             [0. ,       0.660377 , 0.180625 , 0.02935 ,  0.01375  ],
#             [0. ,       0.755241,  0.4153125 ,0.028302,  0.015625 ],
#             [0.  ,      0.806604,  0.41625,   0.028302,  0.01375  ]])
#        image = Image.open('../1TrainData/00058ec5c3f3274f.jpg')
        
        w, h = image.size[:2]

        # after np array, img shape become: h, w, c
        pix = np.array(image)

        # transform the image to rgb if the img is greyscale
        if len(pix.shape) == 2:
            pix = color.gray2rgb(pix)

        # random x min max, y min max thus new width and height of the new img
        new_img_xmin = np.random.uniform(high=self.jitter)
        new_img_xmax = 1-np.random.uniform(high=(self.jitter-new_img_xmin))
        new_img_ymin = np.random.uniform(high=(self.jitter))
        new_img_ymax = 1-np.random.uniform(high=(self.jitter-new_img_ymin))
        new_img_w = (new_img_xmax - new_img_xmin)
        new_img_h = (new_img_ymax - new_img_ymin)

        # crop the image with the random initialization above
        try:
            pix = pix[int(new_img_ymin * h):math.ceil(new_img_ymax * h),
                      int(new_img_xmin * w):math.ceil(new_img_xmax * w)]
        except IndexError:
            print(f"rg1: {int(new_img_ymin * h):math.ceil(new_img_ymax * h)}")
            print(f"rg2: {int(new_img_xmin * w):math.ceil(new_img_xmax * w)}")
            print(f"image name : {image}")
            return
        # transform xcentre, y centre, width and height to xy (min max)
        label_xmin = labels[:, 1] - labels[:, 3]/2
        label_ymin = labels[:, 2] - labels[:, 4]/2
        label_xmax = labels[:, 1] + labels[:, 3]/2
        label_ymax = labels[:, 2] + labels[:, 4]/2

        # new label min coord needs to be in btw 0 and 1
        new_label_x_min = np.clip((label_xmin - new_img_xmin)/new_img_w,
                                  a_min=0.0, a_max=1)
        new_label_y_min = np.clip((label_ymin - new_img_ymin)/new_img_h,
                                  a_min=0.0, a_max=1)
        new_label_w = (label_xmax - new_img_xmin)/new_img_w - new_label_x_min
        new_label_h = (label_ymax - new_img_ymin)/new_img_h - new_label_y_min

#        draw_boxes(pix, labels)
        longer_side = np.max(pix.shape[:2])
        shorter_side = np.min(pix.shape[:2])

        # we used the length ratio to adjust new labels after resizing
        ratio = shorter_side / longer_side
        offset = ((longer_side - shorter_side) / longer_side) / 2

        # adjust the shorter side for resizing
        if np.argmax(pix.shape[:2]):
            labels[:, 4] = new_label_h * ratio
            labels[:, 2] = new_label_y_min*ratio + offset + labels[:, 4]/2
            labels[:, 3] = new_label_w
            labels[:, 1] = new_label_x_min + new_label_w/2
        else:
            labels[:, 3] = new_label_w * ratio
            labels[:, 1] = new_label_x_min*ratio + offset + labels[:, 3]/2
            labels[:, 4] = new_label_h
            labels[:, 2] = new_label_y_min + new_label_h/2

        # make labels that completely cropped out 0
        labels[np.where((labels[:, 3] < 0) | (labels[:, 4] < 0))] = 0

        # resize image according to its orignial aspect ratio
        pix = letterbox_image(pix, (self.inp_dim, self.inp_dim))
#        draw_boxes(pix, labels)
        image = Image.fromarray(pix.astype('uint8'))
        return image, labels
