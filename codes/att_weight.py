import sys
import datetime
import os
import pandas as pd

import torch
from torch.optim import lr_scheduler
from torch import optim
import torch.nn as nn

from config_parser import Config
from model import Model
from data_provider import DataProvider

from seq_encoding import ENCODING_METHOD_MAP


def main():
    # parse config
    config_file = sys.argv[1]

    config = Config(config_file)
    folder = config_file.split('/')[0]
    encoding_func = ENCODING_METHOD_MAP[config.encoding_method]
    encoding_func2= ENCODING_METHOD_MAP[config.encoding_method2]
    data_provider = DataProvider(
        encoding_func,
        encoding_func2,
        config.data_file,
        config.test_file,
        config.batch_size,
        max_len_hla=config.max_len_hla,
        max_len_pep=config.max_len_pep,
        model_count=config.model_count
        )
    device = config.device
    models = config.model_count*config.base_model_count
    print(models)
    total_df=pd.DataFrame()
    for i in range(models):
    # load and prepare model
        path = folder + "/best_model_{}.pytorch".format(i)
        state_dict = torch.load(path)
        model = Model(config)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        data_provider.new_epoch()

        for _ in range(data_provider.test_steps()):
            data = data_provider.batch_test()
            hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, uid_list = data  
            temp_attn = {}
            temp_attn_hla = {}
            with torch.no_grad():
                 temp = model.encoder_peptide2.conv_0(pep2.to(device))
                 temp, att = model.encoder_peptide2.att_0(temp.to(device))

            for i in range(config.batch_size):
                temp_attn[uid_list[i].split('-')[3]]=att[i].tolist()

            temp_df=pd.DataFrame.from_dict(temp_attn,orient="index")

            total_df=pd.concat([total_df,temp_df])

    avg_= total_df.mean(axis=0)
    avg_= pd.DataFrame({'position':avg_.index+1, 'avg weight':avg_.values})  
    avg_.to_csv(folder + "/" + "attn_weight.csv",index=None)

if __name__ == '__main__':

    main()