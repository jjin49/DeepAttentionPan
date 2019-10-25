import sys
import datetime
import os
import pandas as pd

import torch
from torch.optim import lr_scheduler
from torch import optim
import torch.nn as nn

from config_parser import Config
from model import (
    Model,
    weight_initial,
    count_parameters,
)
from data_provider import DataProvider
from logger import (
    setup_logging,
    log_to_file,
)
from callbacks import (
    ModelCheckPointCallBack,
    EarlyStopCallBack,
)
from result_writer import (
    weeekly_result_writer,
    write_metrics_file,
)
from seq_encoding import ENCODING_METHOD_MAP


#############################################################################################
#
# Test
#
#############################################################################################

def batch_test(model, device, data, config):
    hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, uid_list = data
    pred_ic50 = model(hla_a.to(device), hla_mask.to(device), hla_a2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device), pep2.to(device), pep_mask2.to(device))
    return pred_ic50, uid_list

def test(config, data_provider):

    if not config.do_test:
        log_to_file('Skip testing', 'Not enabled testing')
        return

    device = config.device
    
    temp_list=[] 
    for p in range(config.base_model_count):
        for k in range(config.model_count):
    # load and prepare model
             state_dict = torch.load(config.model_save_path(p*config.model_count+k))
             model = Model(config)
             model.load_state_dict(state_dict)
             model.to(device)
             model.eval()
             temp_dict={}
             data_provider.new_epoch()
             for _ in range(data_provider.test_steps()):
                   data = data_provider.batch_test()
                   with torch.no_grad():
                        pred_ic50, uid_list= batch_test(model, device, data, config)
                        for i, uid in enumerate(uid_list):
                            temp_dict[uid] = pred_ic50[i].item()
             temp_list.append(temp_dict)

    # average score of the emsemble model
    result_dict=temp_list[0]
    if config.model_count>1:
       for k in range(1,config.model_count):
           for j in result_dict.keys():
                result_dict[j]+=temp_list[k][j]

    if config.base_model_count>1:
       for p in range(1,config.base_model_count):
           for k in range(config.model_count):
               for j in result_dict.keys():
                   result_dict[j]+=temp_list[p*config.model_count+k][j]

    for j in result_dict.keys():
    	result_dict[j]=result_dict[j]/(config.model_count*config.base_model_count)

    # print(result_dict)
    result_file = weeekly_result_writer(result_dict, config)
    log_to_file('Testing result file', result_file)

    metric_file = write_metrics_file(result_file, config)
    log_to_file('Testing metric result file', metric_file)


#############################################################################################
#
# Main
#
#############################################################################################

def main():
    # parse config
    config_file = sys.argv[1]
    config = Config(config_file)

    # setup logger
    setup_logging(config.working_dir)

    # encoding func
    encoding_func = ENCODING_METHOD_MAP[config.encoding_method]
    encoding_func2= ENCODING_METHOD_MAP[config.encoding_method2]
    log_to_file('Encoding method2', config.encoding_method2)

    data_provider=[]
    for p in range(config.base_model_count):
        temp_provider = DataProvider(
             encoding_func,
             encoding_func2,
             config.data_file,
             config.test_file,
             config.batch_size,
             max_len_hla=config.max_len_hla,
             max_len_pep=config.max_len_pep,
             model_count=config.model_count
        )
        data_provider.append(temp_provider)
 
    log_to_file('max_len_hla', data_provider[0].max_len_hla)
    log_to_file('max_len_pep', data_provider[0].max_len_pep)
    
    test(config, data_provider[0])


if __name__ == '__main__':
    main()
