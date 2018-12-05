from sklearn.model_selection import StratifiedShuffleSplit
from utilis import move_images, prep_labels, load_classes
from train import main
import random
import numpy as np
import pandas as pd
import pickle
import shutil
import os


def move_not_zero(labels_df, label_name):
    labels_df = pd.read_csv(label_name)
    labels_df = pd.unique(labels_df.iloc[:, 0][labels_df.iloc[:, 1] != 0])
    return labels_df


def split_train(label_name, train_size, random_state=0):
    # split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-train_size),
                                 random_state=random_state)
    labels_df = pd.read_csv(label_name)
    print(f"items per class in the original dataframe: \
          \n{labels_df.groupby('c').size()}")
    # get split index
    train_index, others_index = next(iter(
                                   sss.split(labels_df.img_name, labels_df.c)))
    print(f"number of training images before removing duplicates image names: \
          \n{np.size(train_index)}")
    print(f"number of other images before removing duplicates image names: \
          \n{np.size(others_index)}")

    # train dataframe with possible dupliocates
    train_temp = labels_df.iloc[train_index, :]
    print(f"items per class before removing duplicates image names:\
          \n{train_temp.groupby('c').size()}")

    # remove duplicates image names
    train_unique = pd.unique(train_temp.img_name)
    print(f"number of training images after removing duplicate image names: \
          \n{len(train_unique)}")

    # new train dataframe with unique images names
    train = labels_df[labels_df.img_name.isin(train_unique)]
    print(f"items per class with unique image names: \
          \n{train.groupby('c').size()}")
    print(f"Sampled {len(train)} images out of {len(labels_df)} images")
    return train


# label_name = '../1TestData/label.csv'

def compare():
    compare_path = "../5Compare/"
    results_save_name = "../5Compare/results.p"
    to_path_list = {"../250_to_300_imgs/": 1/6,
                    "../400_to_450_imgs/": 1/4,
                    "../550_to_600_imgs/": 3/7}
    classes = load_classes('../4Others/color_ball.names')
    results = {}
    conf_list = np.arange(start=0.1, stop=0.95, step=0.025)
    for to_path, train_size in to_path_list.items():
        file_name = to_path.strip(".").strip("/")
        results[file_name] = {}
        results[file_name]["best_ap"] = {}
        results[file_name]["specific_conf_ap"] = {}
        for cls in classes:
            results[file_name]["best_ap"][cls] = []
            results[file_name]["specific_conf_ap"][cls] = []
        results[file_name]["best_map"] = []
        results[file_name]["best_conf"] = []
        results[file_name]["specific_conf_map"] = []
        for index, seed in enumerate(range(2, 10)):
            random.seed(seed)
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            name_list = ["img_name", "c", "gx", "gy", "gw", "gh"]
            # Original label names
            label_csv_mame = '../color_balls/label.csv'
            img_txt_path = "../color_balls/*.txt"
            prep_labels(img_txt_path, name_list, label_csv_mame)
            # sub sampled label names
            sub_sample_csv_name = to_path + "label.csv"
            sub_sample_txt_path = to_path + "*.txt"
            prep_labels(sub_sample_txt_path, name_list, sub_sample_csv_name)
            # label_csv_mame = '../1TestData/label.csv'
            # img_txt_path = "../1TestData/*.txt"
            move_images(label_name=label_csv_mame, to_path=to_path,
                        action_fn=split_train, train_size=train_size,
                        random_state=seed)
            best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
                map_frame = main(classes, conf_list, sub_sample_csv_name,
                                 sub_sample_txt_path, to_path, cuda=True,
                                 specific_conf=0.5)
            results[file_name]["best_map"].append(best_map)
            results[file_name]["best_conf"].append(best_conf)
            results[file_name]["specific_conf_map"].append(specific_conf_map)
            for cls in classes:
                results[file_name]["best_ap"][cls].append(best_ap)
                results[file_name]["specific_conf_ap"][cls].append(
                        specific_conf_ap)
            map_frame.to_csv(compare_path + file_name + str(index) + ".csv",
                             index=True)
            shutil.rmtree(to_path)
    with open(results_save_name, "wb") as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    compare()

#with open(results_save_name, 'rb') as fp:
#    b = pickle.load(fp)


# code thhat used to recover data
#labels_df = pd.read_csv(label_name)
#labels_df['img_name'] = labels_df['img_name'].str.replace(
#        '1TrainData', 'color_balls')
#img_name = pd.unique(labels_df.iloc[:, 0][labels_df.iloc[:, 1] != 0])
#try_img = img_name[0]
#for images in img_name:
#            shutil.copy(images, to_path)
#
#
#for img in img_name:
#    try_df = labels_df[labels_df["img_name"] == img]
#    try_df = try_df.iloc[:,1:]
#    xx = np.array(try_df)
#    xx[:,0] = np.int_(xx[:,0])
#    xx[:,0] = xx[:,0].astype(int)
#    with open(img[:-3] + 'txt','w') as f:
#        for items in xx:
#            for index, item in enumerate(items):
#                if not index:
#                    f.write(str((int(item))) +" ")
#                else:
#                    f.write(str(item) +" ")
#            f.write("\n")
#IMG_20181106_160325
