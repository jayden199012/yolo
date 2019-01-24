
import tensorflow as tf
import os
import numpy as np
import glob
import pandas as pd


def keep_important_weights_only(rootdir, csv_rootdir, tfevent_rootdir):
    '''
    Results from csv gives the latest results while results from tfevents
    gave us best results, if if the file is either one of them, it will be kept
    otherwise deleted.
    Therefore, please update both csv before executing this code!
    '''
    answer = ''
    while answer not in ['y', 'n']:
        answer = input("This operation might remove some of your " +
                       "important documents, are you sure you want to " +
                       "CONTINUE? [Y/N]").lower()
        if answer == 'y':
            important_weights = pd.read_csv(
                    csv_rootdir, index_col=0)['weights_path'].tolist()
            important_weights.extend(
                    pd.read_csv(
                                tfevent_rootdir,
                                index_col=0)['weights_path'].tolist())
            for root, subdirs, files in os.walk(rootdir, topdown=False):
                if len(subdirs):
                    for subdir in subdirs:
                        path = f"{root}/{subdir}/*.pth"
                        files = glob.glob(path)
                        if len(files):
                            latest_file = max(files, key=os.path.getctime)
                            for file in files:
                                file = file.replace("\\", "/")
                                if (file not in important_weights) or\
                                        (file != latest_file):
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
                                           'pth_position', 'true_path'])
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
                            max_map = df[::-1].max(axis=1).iloc[0]
                            confidence = np.round(float(
                                              np.argmax(df.loc['mAP', :][::-1],
                                                  axis=1)), 3)
                            results_df.loc[count, 'csv_name'] = file.replace(
                                    '\\', '/')
                            # if weight_dir is given, we will look for
                            # the weight pth file from the weight directory
                            if weight_dir:
                                # get the sub path, usually the expriment name
                                sub_p = root.split('/')[-1]
                                # get the name before the file type
                                sub_f = file.split('\\')[-1].split('.')[0]
                                weights_path = f"{weight_dir+sub_p}/{sub_f}*"
                                print(f"file is {file}")
                                print(f"weights_path is {weights_path}")
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
                                    if v.tag == 'best_conf':
                                        results_['confidence'].append(
                                                v.simple_value)
                                    if v.tag == 'best_map':
                                        results_['map'].append(
                                                v.simple_value)
                                        results_['global_steps'].append(
                                                events.step)
                            try:
                                best_map_pos = np.where(
                                        results_['map'] == np.max(
                                                results_['map']))[-1][-1]
                            except (ValueError, IndexError):
                                continue
                            weight_paths = glob.glob(f"{root}/{subdir}/*.pth")
                            try:
                                weights_path = weight_paths[best_map_pos]
                                results_df.loc[count, 'true_path'] = 1
                            except IndexError:
                                weights_path = weight_paths[-1]
                                results_df.loc[count, 'true_path'] = 0
                            results_df.loc[count,
                                           'weights_path'] =\
                                weights_path.replace('\\', '/')
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


# %%
if __name__ == '__main__':
    
    weight_dir = '../4TrainingWeights/experiment/one_anchor/'
#     csv
    #    rootdir = '../5Compare/'
    csv_rootdir = '../5Compare/one_anchor/'
    get_top_n_results(csv_rootdir, csv=True, weight_dir=weight_dir)


    # from tf events
    tfevent_rootdir = '../4TrainingWeights/experiment/one_anchor/'
    get_top_n_results(tfevent_rootdir, csv=False)
    
    result_csv_name = 'top__results.csv'
    keep_important_weights_only(weight_dir, csv_rootdir+result_csv_name,
                                tfevent_rootdir+result_csv_name)
# %%
#csv = '../5Compare/input_size/2018-12-10_18_13_42.903424/608_seed_426_.csv'
#
#xx = pd.read_csv(csv, index_col=0)
#np.argmax(xx.loc['mAP',:][::-1], axis=1)
#xx[::-1].max(axis=1)[0]
#xx.loc['mAP',:][::-1].max(axis=1)
#xx.max(axis=1).iloc[-1]
#np.round(float(np.argmax(xx.loc['mAP',:][::-1], axis=1)), 3)
#x = [0.0, 0.11076324433088303]
#np.argmax(x[::-1][::-1], axis=0)
#np.where(x==np.max(x))
