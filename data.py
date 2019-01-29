import numpy as np
import math
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utilis import letterbox_image
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
        print(f"img_name : {img_name}")
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

    def __init__(self, csv_file, transform=None):
        """
           Same as CustData but opens image using open cv instead of pillow
        """
        self.label_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.list_IDs = self.label_frame.iloc[:, 0].unique()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        img_name = self.list_IDs[idx]
        image = cv2.imread(img_name, 1)
        labels = self.label_frame.iloc[:, 1:][self.label_frame.iloc[
                :, 0] == img_name].astype('float').values.reshape(-1, 5)
        if self.transform:
            image = self.transform(image)
        samples = {'image': image, 'label': labels}
        return samples


class RandomCrop:

    def __init__(self, jitter=0.2, inp_dim=416):
        self.jitter = jitter
        self.inp_dim = inp_dim

    def __call__(self, image, labels):
        # read imgage using PIL formate (w, h, c)
        w, h = image.size[:2]
        pix = np.array(image)
        if len(pix.shape) == 2:
            pix = color.gray2rgb(pix)
        # random x min max, y min max thus new width and height
        new_xmin = np.random.uniform(high=self.jitter)
        new_xmax = 1-np.random.uniform(high=(self.jitter-new_xmin))
        new_ymin = np.random.uniform(high=(self.jitter))
        new_ymax = 1-np.random.uniform(high=(self.jitter-new_ymin))
        new_w = new_xmax - new_xmin
        new_h = new_ymax - new_ymin

        # new images size should always fit the longer side of the image
        size = np.max([new_w, new_h])

        # difference between new width and height
        off_set = np.abs(new_w-new_h)/size

        # I wrote this try as there was an error happened here before so
        # in case it happens again
        try:
            pix = pix[int(new_ymin * w):math.ceil(new_ymax * w),
                      int(new_xmin * h):math.ceil(new_xmax * h)]
        except IndexError:
            print(f"first int : {int(new_ymin * w):math.ceil(new_ymax * w)}")
            print(f"second int : {int(new_xmin * h):math.ceil(new_xmax * h)}")
            print(f"image name : {image}")
            return

        # add canvas to the new image so that it is in original ratio
        pix = letterbox_image(pix, (self.inp_dim, self.inp_dim))

        # adjust label coordinates to offset the canvas
        adjust_width = 0 if new_w > new_h else 1
        if adjust_width:
            off_set_list = [off_set/2, 0]
        else:
            off_set_list = [0, off_set/2]

        # adjust labels coordinates according to the crop
        for index, label in enumerate(labels):

            label_xmin = label[1] - label[3]/2
            label_ymin = label[2] - label[4]/2
            label_xmax = label[1] + label[3]/2
            label_ymax = label[2] + label[4]/2
            label_w = np.min([(
                    label_xmax - np.max([new_xmin, label_xmin])) / size,
                                0.999])
            label_h = np.min([(
                    label_ymax - np.max([new_ymin, label_ymin])
                    )/size, 0.999])
            x_min = np.max([(label_xmin-new_xmin)/size, 0.0]) + off_set_list[0]
            y_min = np.max([(label_ymin-new_ymin)/size, 0.0]) + off_set_list[1]
            label[1] = x_min + label_w/2
            label[2] = y_min + label_h/2
            label[3] = label_w
            label[4] = label_h

            # zero out labels when the object is completely cropped out
            if any(x for x in label[3:5] < 0) or any(
                    x for x in label[1:3] > 1):
                labels[index] = 0.0
        image = Image.fromarray(pix.astype('uint8'))
        return image, labels
