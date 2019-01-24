from __future__ import division

import torch
from utilis import prep_params
from train import run_training
import itertools as it
import datetime
import os
import time
import pandas as pd


if __name__ == "__main__":
    date_time_now = str(
        datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    compare_path = f"../5Compare/batch_size/{date_time_now}/"
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)

    config = {'label_csv_mame': '../1TrainData/label.csv',
              'img_txt_path': "../1TrainData/*.txt",
              # label csv column names
              'name_list': ["img_name", "c", "gx", "gy", "gw", "gh"],
              'test_label_csv_mame': '../1TestData/label.csv',
              'test_img_txt_path': "../1TestData/*.txt",
              'cfg_path': "../4Others/yolo.cfg",
              'params_dir': '../4Others/params.txt'}

    # turn cudnn on or off depends on situation
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
# =============================================================================
#   experiment
# =============================================================================
    test_name = 'batch_size'
    test_value = [5, 10, 15, 20]

#    test_name = 'input_sizes'
#    test_value = [448, 480]
#
#    test_name = 'conf_lambda_list'
#    input_sizes = list(range(1, 6))
# =============================================================================
#
# =============================================================================

    '''
        you can specify more layered experiments here , but do not use
        np.arrange() as it will result an error for json.dumps()
    '''
    tune_params = {'seed': list(range(424, 428)),
                   test_name: test_value,
                   }

    index, values = zip(*tune_params.items())
    experiments_params = [dict(zip(index, v)) for v in it.product(*values)]
    time_taken_df = pd.DataFrame(columns=tune_params['seed'])
    start = time.time()
    for experiment_params in experiments_params:
        params = prep_params(config['params_dir'],
                             config['label_csv_mame'],
                             experiment_params)
        params['sub_name'] = f"{test_name}_seed_{params['seed']}_"
        x_start = time.time()
        best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap,\
            map_frame = run_training(params=params, **config)
        map_frame.to_csv(f"{compare_path+params['sub_name']}.csv",
                         index=True)
        time_taken_df.loc[params[test_name],
                          params['seed']] = time.time() - x_start
        time_taken_df.to_csv(compare_path + 'time_taken.csv', index=True)
    time_taken = time.time()-start
    print(f"This experiment took {time_taken//(60*60)} hours : \
                                  {time_taken%60} minutes : \
                                  {time_taken%60} seconds!")
