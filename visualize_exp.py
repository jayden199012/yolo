import torch
import time
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import ssaplot
from evaluate import get_map
from torch.utils.data import DataLoader
from data import CustData
from torchvision import transforms
from utilis import parse_cfg,  my_collate, load_classes
from yolo_v3 import yolo_v3
import warnings
warnings.filterwarnings("ignore")


class results():
    def __init__(self, results_path, test_name_list, csv_name, figsize=(20, 8),
                 title_size=20, hspace=0.6, wspace=0.4):
        self.results_path = results_path
        self.csv_name = csv_name
        self.test_name_list = test_name_list
        self.all_df = self.get_avg_df()
        self.map = self.get_map_df()
        self.figsize = figsize
        self.title_size = title_size
        self.hspace = hspace
        self.wspace = wspace
        self.time_mean = self.get_time_mean()

    def get_time_mean(self):
        df = pd.read_csv(f"{self.results_path}time_taken.csv", index_col=0)
        time_mean = df.mean(axis=1)
        return time_mean

    def get_avg_df(self):
        df = {}
        for item in self.test_name_list:
            test_name = self.results_path + item + "*.csv"
            read_files = glob.glob(test_name)
            count = 0
            for file in read_files:
                df_ = pd.read_csv(file, index_col=0)
                df_.columns = [np.round(np.float(x), 3) for x in df_.columns]
                if not count:
                    df[item] = df_
                else:
                    df[item] += df_
                count += 1
            df[item] = df[item]/count
        return df

    def get_map_df(self):
        columns = list(self.all_df.values())[0].columns.values
        index = list(self.all_df.keys())
        map_df = pd.DataFrame(index=index, columns=columns)
        for index, value in enumerate(self.all_df.values()):
            map_df.iloc[index, :] = value.iloc[4, :].values
        return map_df

    def kdp_all(self, df, seperate=False):
        plt.figure(figsize=self.figsize)
        plt.xlim(0, 1)
        plt.xlabel("mAP", fontsize=20)
        plt.ylabel("Kernel Density", fontsize=20)
        for i in range(len(df)):
            data = df.iloc[i, :]
            name = df.iloc[i, :].name
            color = self.color(df, i)
            sns.kdeplot(data, label=name, color=color, shade=True)
            plt.legend(loc='upper right')
        plt.xlabel("Object Confidence", fontsize=20)
        plt.ylabel("Knernel Density", fontsize=20)

    def kdp_seperate(self, df, col=1):
        fig = plt.figure(figsize=self.figsize)
        fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)
        for i in range(len(df)):
            data = df.iloc[i, :].values
            name = df.iloc[i, :].name
            color = self.color(df, i)
            rows = np.ceil(len(df) / col)
            ax = fig.add_subplot(rows, col, i+1)
            sns.distplot(data, color=color, kde=False)
            ax.set_title(name)
        ax.set_xlabel("mAP", fontsize=20)

    def line_all(self, df, seperate=False):
        plt.figure(figsize=self.figsize)
        for i in range(len(df)):
            data = df.iloc[i, :]
            name = df.iloc[i, :].name
            color = self.color(df, i)
            plt.plot(data, label=name, color=color)
            plt.legend(loc='upper right')
        plt.xlabel("Object Confidence", fontsize=20)
        plt.ylabel("mAP", fontsize=20)

    def line_seperate(self, df, col=1):
        fig = plt.figure(figsize=self.figsize)
        fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)
        for i in range(len(df)):
            data = df.iloc[i, :]
            name = df.iloc[i, :].name
            color = self.color(df, i)
            rows = np.ceil(len(df) / col)
            ax = fig.add_subplot(rows, col, i+1)
            ax.plot(data, color=color)
            ax.set_title(name)
            ax.set_xlabel("Object Confidence", fontsize=20)
            ax.set_ylabel("mAP", fontsize=20)

#        fig, ax = subplots(rows, col, figsize=self.figsize)
#        ax.set_xlabel('Xlabel')
#        ax.set_ylabel('Ylabel')
#        ax.set_xlim(0,1)
#

    def best_map(self, df, sort=True, only_map=True, fontsize=30):
        plt.figure(figsize=self.figsize)
        if only_map:
            map_max = df.max(axis=1)
        # when its false, first find out the confidence where its map is the
        # highest, then find the corresponding ap for each class
        else:
            best_conf = df.iloc[4, :].idxmax(axis=1)
            map_max = df[best_conf]
        if sort:
            map_max = map_max.sort_values(ascending=False)
        self.bar(map_max, fontsize)
#        plt.xlabel("Number of Images", fontsize=20)
        plt.ylabel("mAP", fontsize=20)

    def bar(self, df, fontsize):
        ax = sns.barplot(x=df.index, y=df.values, palette=sns.color_palette(
                "RdYlBu", 12), saturation=0.85)
        ssaplot.annotate(ax, message='Float', fontsize=fontsize)

    def compare_vis(self, func, **kwargs):
        for index, value in self.all_df.items():
            func(value, **kwargs)
            plt.suptitle(index, fontsize=self.title_size)
            plt.show()

    def compare_map(self, func, **kwargs):
            func(self.map, **kwargs)
            plt.suptitle("mAP Comparism", fontsize=self.title_size)
            plt.show()

    # modify the color part for your own use case
    def color(self, df, i):
        try:
            color = df.iloc[i, :].name.split('_')[0].replace('mAP', 'black')
        except:
            color = np.random.randint(0, 255)
        return color

    def map_improvement(self):
        plt.figure(figsize=self.figsize)
        # finding the max mAP for all tests
        map_max = self.map.max(axis=1)
        # number of mAP categaries
        len_map = len(map_max.values)
        y = []
        x = []
        for i in range(1, len_map):
            improvement = np.log(map_max.values[i]/map_max.values[i-1]) * 100
            index = f"From {map_max.index[i-1]} to {map_max.index[i]}"
            y.append(improvement)
            x.append(index)
        ax = sns.barplot(x=x, y=y, palette=sns.color_palette("Paired", 4),
                         saturation=0.85)
        ssaplot.annotate(ax, message='Float', fontsize=30)
#        plt.xlabel("Number of Images", fontsize=20)
        plt.ylabel("mAP percentage change (%)", fontsize=20)
        plt.suptitle("mAP percentage change", fontsize=self.title_size)
        plt.show()

    def heat_map(self, annotation=False):
        if not annotation:
            linewidths = 0
        else:
            linewidths = .3
        data = pd.read_csv(self.results_path + self.csv_name, index_col=0)
        grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        fig, axes = plt.subplots(2, figsize=self.figsize, gridspec_kw=grid_kws)
        ax1 = sns.heatmap(data.round(3), ax=axes[0], cbar_ax=axes[1],
                          cbar_kws={"orientation": "horizontal"},
                          linewidths=linewidths, annot=annotation)
        ax1.invert_yaxis()
        axes[0].set_xlabel("object Confidence", fontsize=20)
        axes[0].set_ylabel("IoU Threshhold", fontsize=20)
        axes[1].set_xlabel("Mean Average Precision (mAP) values", fontsize=15)

    def get_time(self, label_fontsize=20, units='seconds'):
        plt.figure(figsize=self.figsize)
        if units == 'seconds':
            time_mean = round(self.time_mean, 2)
        elif units == 'minutes':
            time_mean = round(self.time_mean / 60, 2)
        elif units == 'hours':
            time_mean = round(self.time_mean / 360, 2)
        else:
            raise NameError(f"Effective units are 'seconds', 'minutes' and \
                            'hours'")
        self.bar(time_mean, label_fontsize)
        plt.ylabel(f"Average Training Time Taken in ({units})", fontsize=20)

    def time_increase(self):
        plt.figure(figsize=self.figsize)
        x = []
        y = []
        for i in range(1, len(self.time_mean)):
            percentage_change = np.log(
                    self.time_mean.values[i]/self.time_mean.values[i-1]
                                ) * 100
            index = f"From input size : {self.time_mean.index[i-1]} " +\
                    f"to {self.time_mean.index[i]}"
            y.append(percentage_change)
            x.append(index)
        ax = sns.barplot(x=x, y=y, palette=sns.color_palette("Paired", 4),
                         saturation=0.85)
        ssaplot.annotate(ax, message='Float', fontsize=30)
#        plt.xlabel("Number of Images", fontsize=20)
        plt.ylabel("Training time percentage change", fontsize=20)
        plt.suptitle("Training time percentage change",
                     fontsize=self.title_size)
        plt.show()


def show():
    results_path = "../5Compare/img_size/2018-12-08_21_45_38.145340/"
#    results_path = "../5Compare/input_size/2018-12-09_00_25_41.640054/"
    csv_name = "con_iou_map_frame.csv"
    test_name_list = ["250_to_300_imgs", "400_to_450_imgs", "550_to_600_imgs",
                      "700_to_750_imgs"]
#    test_name_list = ['608', '512', '416', '320']
    visual = results(results_path, test_name_list, csv_name)
    visual.compare_vis(visual.best_map)
    visual.map_improvement()
    visual.figsize = (14, 8)
    visual.compare_vis(visual.best_map, only_map=False)
    visual.compare_vis(visual.line_all)
    visual.compare_map(visual.line_all)
    visual.compare_map(visual.best_map)
    visual.compare_vis(visual.kdp_seperate)
    visual.compare_vis(visual.kdp_all)
    visual.heat_map()
    visual.get_time(label_fontsize=25, units='minutes')
    visual.time_increase()
    visual.heat_map(annotation=True)
#    con_iou_map_frame(results_path, csv_name)


def conf_map_frame(iou_conf, conf_list):
    cuda = True
    specific_conf = 0.9
    cfg_path = "../4Others/color_ball.cfg"
    test_root_dir = "../1TestData"
    test_label_csv_mame = '../1TestData/label.csv'
    classes = load_classes('../4Others/color_ball.names')
    blocks = parse_cfg(cfg_path)
    model = yolo_v3(blocks)
    checkpoint_path = "../4TrainingWeights/experiment/2018-11-30_05_19_48.404150/2018-11-30_05_59_33.719706_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    test_transform = transforms.Compose(
            [transforms.Resize(model.net["height"]), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    test_data = CustData(test_label_csv_mame, test_root_dir,
                         transform=test_transform)

    test_loader = DataLoader(test_data, shuffle=False,
                             batch_size=model.net["batch"],
                             collate_fn=my_collate,
                             num_workers=6)
    best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
        map_frame = get_map(model, test_loader, cuda, conf_list, iou_conf,
                            classes, False, specific_conf, True)
    return best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
        map_frame


def con_iou_map_frame(compare_path, csv_name):
    start = time.time()
    # object confidence level
    conf_list = np.arange(start=0.2, stop=0.95, step=0.025)
    # IoU confidence level for the model determine a True positive
    iou_conf_list = np.arange(start=0.2, stop=0.95, step=0.025)
    con_iou_map_frame = pd.DataFrame(index=iou_conf_list, columns=conf_list)
    for iou_conf in iou_conf_list:
        print(f"Running for IoU : {iou_conf}")
        outputs = conf_map_frame(iou_conf, conf_list)
        map_frame = outputs[-1]
        con_iou_map_frame.loc[iou_conf, :] = map_frame.iloc[4, :]
    con_iou_map_frame.index = np.round(con_iou_map_frame.index, 3)
    con_iou_map_frame.columns = np.round(con_iou_map_frame.columns, 3)
    con_iou_map_frame.to_csv(compare_path + csv_name, index=True)
    time_taken = time.time()-start
    print(f"This experiment took {time_taken//(60*60)} hours : \
                                  {time_taken//60} minutes : \
                                  {time_taken%60} seconds!")


if __name__ == '__main__':
        compare_path = "../5Compare/"
        csv_name = "con_iou_map_frame.csv"
        con_iou_map_frame(compare_path, csv_name)
