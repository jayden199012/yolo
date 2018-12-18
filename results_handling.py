
import tensorflow as tf
import os
import numpy as np
import glob
import pandas as pd



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
    # count is used to locate the rows for the output data frame
    count = 0
    if csv:
        results_df = pd.DataFrame(columns=['csv_name', 'weights_path',
                                           'max_mAP', 'confidence'])
    else:
        results_df = pd.DataFrame(columns=['weights_path', 'max_mAP',
                                           'global_step', 'confidence',
                                           'pth_position'])
    for root, subdirs, files in os.walk(rootdir, topdown=False):
        if len(subdirs):
            for subdir in subdirs:
                # pulling information from csv files
                if csv:
                    path = f"{root}/{subdir}/*_.csv"
                # pulling information from tf event file
                else:
                    path = f"{root}/{subdir}/events.out.tfevents*"
                files = glob.glob(path)
                if len(files):
                    for file in files:
                        # handle from output csv
                        if csv:
                            df = pd.read_csv(file, index_col=0)
                            max_map = df.max(axis=1).iloc[-1]
                            confidence = np.round(float(
                                                    df.idxmax(1).iloc[-1]), 3)
                            
                            results_df.loc[count, 'csv_name'] = file
                            # if weight_dir is given, we will look for
                            # the weight pth file from the weight directory
                            if weight_dir:
                                # get the sub path, usually the expriment name
                                sub_p = root.split('/')[-1]
                                # get the name before the file type
                                sub_f = file.split('.')[0]
                                weights_path = f"{weight_dir+sub_p}/{sub_f}*"
                                weights_path = glob.glob(weights_path)[-1]
                                weights_path = glob.glob(
                                        weights_path+"/*.pth")[-1]
                                results_df.loc[count, 'weights_path'] =\
                                    weights_path.replace('\\', '/')            
                        # handle the tf events
                        else:
                            results_ = {'map': [],
                                        'confidence': [],
                                        'position': [],
                                        'global_steps': []}
                            print(f"this is file :{file}")
                            for events in tf.train.summary_iterator(file):
                                for v in events.summary.value:
                                    print(f"all v tags :{v.tag}")
                                    if v.tag == 'best_conf':
                                        results_['confidence'].append(
                                                v.simple_value)
                                    if v.tag == 'best_map':
                                        print(f"best map {v.simple_value}")
                                        results_['map'].append(
                                                v.simple_value)
                                        results_['global_steps'].append(
                                                events.step)
                            try:
                                best_map_pos = np.argmax(results_['map'])
                            except ValueError:
                                continue
                            results_df.loc[count,
                                           'weights_path'] = file
                            results_df.loc[count,
                                           'pth_position'] = best_map_pos
                            results_df.loc[count,
                                           'global_step'] =\
                                results_['global_steps'][best_map_pos]
                            max_map = np.max(results_['map'])
                            confidence = results_['confidence'][
                                         best_map_pos]
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

# csv
rootdir = '../5Compare/'

# from tf events
rootdir = '../4TrainingWeights/experiment/'

weight_dir = '../4TrainingWeights/experiment/'

get_top_n_results(rootdir, n=100, csv=True, weight_dir=weight_dir)

