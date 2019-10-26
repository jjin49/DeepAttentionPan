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

import math

from seq_encoding import ENCODING_METHOD_MAP


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


#############################################################################################
#
# Main
#
#############################################################################################



def main():

    config =Config("dup_0/config.json")

    # get HLA and peptide
    HLA = sys.argv[1]
    peptide = sys.argv[2]
    if len(peptide) > 15:
       print("please input the peptide shorter than 16 amino acids.")
       return

    hla_seq = 0
    # get the sequence of HLA
    hla_path = os.path.join(BASE_DIR, '..','dataset',  'mhc_i_protein_seq2.txt')    
    with open(hla_path, 'r') as in_file:
         for line_num, line in enumerate(in_file):
             if line_num == 0:
                continue

             info = line.strip('\n').split(' ')
             if info[0] != HLA:
                continue
             hla_seq = info[1]
             break
    if hla_seq == 0:
       print("The HLA is not included  in the dataset.")
       return

    # encode the sequences

    encoding_func = ENCODING_METHOD_MAP["one_hot"]
    encoding_func2= ENCODING_METHOD_MAP["blosum"]

    hla, hla_mask = encoding_func(hla_seq, 385)
    hla = torch.reshape(hla,(1,hla.size(0), hla.size(1)))

    pep, pep_mask = encoding_func(peptide, 15)
    pep = torch.reshape(pep,(1,pep.size(0), pep.size(1)))

    hla2, hla_mask2 = encoding_func2(hla_seq, 385)
    hla2 = torch.reshape(hla2,(1,hla2.size(0), hla2.size(1)))
    pep2, pep_mask2 = encoding_func2(peptide, 15)      
    pep2 = torch.reshape(pep2,(1,pep2.size(0), pep2.size(1)))

    
    # load model
    device = config.device
 
    temp_list=[] 
    for p in range(config.base_model_count):
        for k in range(config.model_count):
             state_dict = torch.load(config.model_save_path(p*config.model_count+k))
             model = Model(config)
             model.load_state_dict(state_dict)
             model.to(device)
             model.eval()    
             with torch.no_grad():
                  pred_ic50 = model(hla.to(device), hla_mask.to(device), hla2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device), pep2.to(device), pep_mask2.to(device))
                  pred_ic50 = math.pow(50000,1-pred_ic50)


             temp_list.append(pred_ic50)

    pred_ic50 = sum(temp_list)/len(temp_list)          

    print("the predicted IC50 value is : {}".format(pred_ic50))


if __name__ == '__main__':
    main()
