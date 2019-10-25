# -*- coding: utf-8 -*-

import os
import json

import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    def __init__(self, json_file):
        self.config = json.loads(open(json_file).read())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")


    @property
    def max_len_hla(self):
        return self.config['Data']['max_len_hla']

    @property
    def max_len_pep(self):
        return self.config['Data']['max_len_pep']

    @property
    def batch_size(self):
        return self.config['Training']['batch_size']

    @property
    def working_dir(self):
        return os.path.join(BASE_DIR, self.config['Paths']['working_dir'])

    @property
    def data_file(self):
        return os.path.join(BASE_DIR, '..',  'dataset', self.config['Data']['data_file'])

    @property
    def test_file(self):
        return os.path.join(BASE_DIR, '..',  'dataset', self.config['Data']['test_file'])


    @property
    def model_config(self):
        return self.config['Model']

    @property
    def grad_clip(self):
        return self.config['Training']['grad_clip']

    @property
    def start_lr(self):
        return self.config['Training']['start_lr']

    @property
    def min_lr(self):
        return self.config['Training']['min_lr']

    @property
    def epochs(self):
        return self.config['Training']['epochs']

    @property
    def loss_delta(self):
        return self.config['Training']['loss_delta']

    @property
    def encoding_method(self):
        return self.model_config['encoding_method']

    @property
    def encoding_method2(self):
        return self.model_config['encoding_method2']    

    @property
    def do_train(self):
        return self.config['do_train']

    @property
    def do_test(self):
        return self.config['do_test']

    @property
    def model_count(self):
        return self.config['model_count']

    def model_save_path(self,value):
        return os.path.join(self.working_dir, 'best_model_{}.pytorch'.format(value))
    
    @property
    def base_model_count(self):
        return self.config['base_model_count']

    def model_base_path(self,value):
        return os.path.join(self.working_dir, 'base_model_{}.pytorch'.format(value))