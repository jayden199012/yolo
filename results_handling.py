
import tensorflow as tf
import os
import numpy as np
import glob
import pandas as pd
from PIL import Image
from statistics import mode
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import Launch_Functions as lf


def keep_latest_weights_only(rootdir):
    answer = ''
    while answer not in ['y', 'n']:
        answer = input("This operation might remove some of your " +
                       "important documents, are you sure you want to " +
                       "CONTINUE? [Y/N]").lower()
        if answer == 'y':
            for root, subdirs, files in os.walk(rootdir, topdown=False):
                if len(subdirs):
                    for subdir in subdirs:
                        path = f"{root}/{subdir}/*.pth"
                        files = glob.glob(path)
                        if len(files):
                            best_file = max(files, key=os.path.getctime)
                            for file in files:
                                if file != best_file:
                                    os.remove(file)
                        else:
                            try:
                                os.rmdir(subdir)
                            except FileNotFoundError:
                                continue
        elif answer == 'n':
            break
        else:
            print("Invalid input!")
            continue


def get_top_n_results(rootdir, n="", csv=True, weight_dir=''):
    count = 0
    if csv:
        results_df = pd.DataFrame(columns=['csv_name', 'weights_path',
                                           'max_mAP', 'confidence'])
    else:
        results_df = pd.DataFrame(columns=['weights_path',
                                           'max_mAP', 'confidence'])
    for root, subdirs, files in os.walk(rootdir, topdown=False):
        if len(subdirs):
            for subdir in subdirs:
                if csv:
                    path = f"{root}/{subdir}/*_.csv"
                else:
                    path = f"{root}/{subdir}/events.out.tfevents*"
                files = glob.glob(path)
                if len(files):
                    for file in files:
                        if csv:
                            df = pd.read_csv(file, index_col=0)
                            max_map = df.max(axis=1).iloc[-1]
                            confidence = np.round(float(
                                                    df.idxmax(1).iloc[-1]), 3)
                            
                            results_df.loc[count, 'csv_name'] = file
                            if weight_dir:
                                sub_p = root.split('/')[-1]
                                sub_f = file.split('.')[0]
                                weights_path = f"{weight_dir+sub_p}/{sub_f}*"
                                print(weights_path)
                                weights_path = glob.glob(weights_path)[-1]
                                weights_path = glob.glob(
                                        weights_path+"/*.pth")[-1]
                                results_df.loc[count, 'weights_path'] =\
                                    weights_path.replace('\\', '/')
                        results_df.loc[count, 'max_mAP'] = max_map
                        results_df.loc[count, 'confidence'] = confidence
                        count += 1
    results_df = results_df.sort_values(by='max_mAP', ascending=False)
    results_df = results_df.reset_index(drop=True)
    if n:
        results_df = results_df.iloc[:n, :]

    out_name = rootdir+f'top_{n}_results.csv'
    if not os.path.exists(out_name):
        results_df.to_csv(out_name)
    else:
        answer = ''
        while answer not in ['y', 'n']:
            answer = input('WARNING: This file already exists! Do you wish' +
                           ' to CONTINUE? [Y/N]').lower()
        if answer == 'y':
            results_df.to_csv(out_name)
    return results_df


rootdir = '../5Compare/'
rootdir = '../4/input_size/2018-12-10_18_13_42.903424/'
weight_dir = '../4TrainingWeights/experiment/'
rootdir = '../4TrainingWeights/coNf_loss/1_seed_424_*'
files = glob.glob(rootdir)[-1]

get_top_n_results(rootdir, n=10, weight_dir=weight_dir)



for event in tf.train.summary_iterator("../4TrainingWeights/epoch_effect/700_to_750_imgs_seed_422_epoch_35_2018-12-12_10_07_37.617714/events.out.tfevents.1544627258.DESKTOP-TM1BVCG"):
    for v in event.summary.value:
        if v.tag == 'best_conf':
             print(v.simple_value)
        if v.tag == 'best_map':
             print(event.step)
             print(v.simple_value)
l = [12,12,344]
np.argmax(l)
r = '../5Compare/batch_size'

r.split('/')[-1]
