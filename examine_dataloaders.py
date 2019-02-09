from __future__ import division

import numpy as np
import torch
import logging
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from utilis import my_collate, worker_init_fn, ImgToTensorCv,\
    OfssetLabels, draw_boxes, prep_labels
from data import CustData, RandomCrop, CustDataCV


def test_loaders():
    label_csv_mame = '../1TestData/label.csv'
    name_list = ["img_name", "c", "gx", "gy", "gw", "gh"]
    test_img_txt_path = "../1TestData/*.txt"
    prep_labels(test_img_txt_path, name_list, label_csv_mame)
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    # training transformation
    pre_trans = RandomCrop(jitter=0.25,
                           inp_dim=480)
    train_transform = transforms.Compose(
            [transforms.ColorJitter(
                                    brightness=0.25,
                                    saturation=0.25
                                    ),
             transforms.ToTensor()
             ])
    test_pre_trans = OfssetLabels(resize=True, input_dim=480)
    test_transform = transforms.Compose(
               [ImgToTensorCv()])
    train_data = CustData('../1TrainData/label.csv',
                          pre_trans=pre_trans,
                          transform=train_transform)
#    test_data = CustData(test_label_csv_mame,
#                         transform=test_transform)
    test_data = CustDataCV('../1TestData/label.csv',
                           pre_trans=test_pre_trans,
                           transform=test_transform)
    train_loader = DataLoader(train_data, shuffle=False,
                              batch_size=2,
                              collate_fn=my_collate,
                              num_workers=0,
                              worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_data, shuffle=False,
                             batch_size=4,
                             collate_fn=my_collate,
                             num_workers=0,
                             worker_init_fn=worker_init_fn)

#    print("running train loader")
#    for step, samples in enumerate(train_loader):
#        if step < 200:
#            images, labels = samples["image"], samples["label"]
#            for img, label in zip(images, labels):
#                img = img.permute(1, 2, 0).contiguous().numpy()
#                draw_boxes(img, label)
#        else:
#            break

    print("running test loader")
    for step, samples in enumerate(test_loader):
        if step < 2:
            images, labels = samples["image"], samples["label"]
            for img, label in zip(images, labels):
                img = img.permute(1, 2, 0).contiguous().numpy()
                draw_boxes(img, label)
        else:
            break
#img = '../1TrainData/000cf4b56061f60f.jpg'
#img = np.array(Image.open(img))
#img = cv2.imread(img)
#fig, ax = plt.subplots(1, figsize=(8, 8))
#ax.imshow(img)
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # test images
    test_loaders()
