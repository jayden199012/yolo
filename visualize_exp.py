import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", 20)


class results():
    def __init__(self, results_path, test_name_list, figsize=(15, 8),
                 title_size=20, hspace=0.6, wspace=0.4):
        self.results_path = results_path
        self.test_name_list = test_name_list
        self.all_df = self.get_avg_df()
        self.figsize = figsize
        self.title_size = title_size
        self.hspace = hspace
        self.wspace = wspace

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

    def kdp_all(self, df, seperate=False):
        plt.figure(figsize=self.figsize)
        for i in range(len(df)):
            data = df.iloc[i, :]
            name = df.iloc[i, :].name
            # modify the color part for your own use case
            color = df.iloc[i, :].name.split('_')[0].replace('mAP', 'black')
            sns.kdeplot(data, label=name, color=color, shade=True)
            plt.legend(loc='upper right')
 

    def kdp_seperate(self, df, col=1):
        fig = plt.figure(figsize=self.figsize)
        fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)
        for i in range(len(df)):
            data = df.iloc[i, :].values
            name = df.iloc[i, :].name
            # modify the color part for your own use case
            color = df.iloc[i, :].name.split('_')[0].replace('mAP', 'black')
            rows = np.ceil(len(df) / col)
            ax = fig.add_subplot(rows, col, i+1)
            sns.kdeplot(data, color=color, shade=True)
            ax.set_title(name)


    def line_all(self, df, seperate=False):
        plt.figure(figsize=self.figsize)
        for i in range(len(df)):
            data = df.iloc[i, :]
            name = df.iloc[i, :].name
            # modify the color part for your own use case
            color = df.iloc[i, :].name.split('_')[0].replace('mAP', 'black')
            plt.plot(data, label=name, color=color)
            plt.legend(loc='upper right')


    def line_seperatel(self, df, col=1):
        fig = plt.figure(figsize=self.figsize)
        fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)
        for i in range(len(df)):
            data = df.iloc[i, :]
            name = df.iloc[i, :].name
            # modify the color part for your own use case
            color = df.iloc[i, :].name.split('_')[0].replace('mAP', 'black')
            rows = np.ceil(len(df) / col)
            ax = fig.add_subplot(rows, col, i+1)
            ax.plot(data, color=color)
            ax.set_title(name)
#

    def best_map(self, df, sort=True):
        plt.figure(figsize=self.figsize)
        best_conf = df.iloc[4, :].idxmax(axis=1)
        if sort:
            mAP = df[best_conf].sort_values(ascending=False)
        else:
            mAP = df[best_conf]
        sns.barplot(x=mAP.values, y=mAP.index, palette=sns.color_palette(
                "RdYlBu", 10), saturation=0.7)


    
    def compare_vis(self, func):
        for index, value in self.all_df.items():
            func(value)
            plt.suptitle(index, fontsize=self.title_size)
            plt.show()
            


    
results_path = "../5Compare/"
test_name_list = ["250_to_300", "400_to_450", "550_to_600"]
visual = results(results_path, test_name_list)
#print(visual.all_df.values())
#visual.compare_vis(visual.line_seperatel)
visual.compare_vis(visual.best_map)
#visual.kdp_seperate()
#visual.line_seperate()
#visual.line_all()
#
#df = pd.read_csv(r'../5Compare\550_to_600_imgs0.csv', index_col=0)
#df.columns = [np.round(np.float(x), 3) for x in df.columns]

#df2 = pd.read_csv(r'../5Compare\550_to_600_imgs1.csv', index_col=0)
#df3 = df + df2
#df3 = df3 / 2
#
#
#for item in test_name_list:
#    test_name = results_path + item + "*.csv"
#    read_files = glob.glob(test_name)
#    count = 0
#    for file in read_files:
#        df = pd.read_csv(file, index_col=0)
#        if not count:
#            df_ = df
#        else:
#            df_ += df 
#        count += 1
#def kdp_all(df):
#    plt.figure(figsize=(15, 8))
#    for i in range(len(df)):
#        data = df.iloc[i, :]
#        name = df.iloc[i, :].name
#        color = df.iloc[i, :].name.split('_')[0].replace('mAP', 'black')
#        sns.kdeplot(data, label=name, color=color, shade=True)
#        plt.legend(loc='upper left')
#    plt.show()
#
#
#def kdp_seperatel(df, figsize, col=2):
#    fig = plt.figure(figsize=figsize)
#    fig.subplots_adjust(hspace=0.4, wspace=0.4)
#    for i in range(len(df)):
#        data = df.iloc[i, :].values
#        name = df.iloc[i, :].name
#        color = df.iloc[i, :].name.split('_')[0].replace('mAP', 'black')
#        rows = np.ceil(len(df) / col)
#        ax = fig.add_subplot(rows, col, i+1)
#        ax.plot(data, color=color)
#        ax.set_title(name)
#    plt.show()
#plt.figure(figsize=(15, 8))
#plt.plot(df.iloc[i, :].values)
#
