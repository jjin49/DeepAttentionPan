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

from seq_encoding import ENCODING_METHOD_MAP

#############################################################################################
#
# Train
#
#############################################################################################

def batch_train(model, device, data, config):
    hla_a, hla_mask, hla_a2, hla_mask2,  pep, pep_mask, pep2, pep_mask2, ic50,samples = data

    pred_ic50= model(hla_a.to(device), hla_mask.to(device), hla_a2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device),pep2.to(device), pep_mask2.to(device))
    loss = nn.MSELoss()(pred_ic50.to(config.cpu_device), ic50.view(ic50.size(0), 1))

    return loss


def batch_validation(model, device, data, config):
    hla_a, hla_mask, hla_a2, hla_mask2,  pep, pep_mask, pep2, pep_mask2, ic50,samples = data
    with torch.no_grad():
    	 # validation_call
         pred_ic50= model(hla_a.to(device), hla_mask.to(device), hla_a2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device),pep2.to(device), pep_mask2.to(device))
         loss = nn.MSELoss()( pred_ic50.to(config.cpu_device), ic50.view(ic50.size(0), 1))
         pred_ic50=pred_ic50.view(len(pred_ic50)).tolist()
         # print(pred_ic50)
         # print("pred_ic50_len:{}".format(len(pred_ic50)))
         # print(ic50)
         # print("ic50_len:{}".format(len(ic50)))
         # print(samples)
         # print("samples_len:{}".format(len(samples)))
         # exit()

         return loss,pred_ic50,samples


def train(config, data_provider,p):
    # skip training if test mode
    if not config.do_train:
       log_to_file('Skip train', 'Not enabled training')
       return
    device = config.device
    log_to_file('Device', device)
        # log pytorch version
    log_to_file('PyTorch version', torch.__version__)
        # prepare model
    log_to_file('based on base_model #', p)

    for i in range(config.model_count):
        log_to_file('begin training model #',i)
        model = Model(config)
        weight_initial(model, config)
        model.to(device)
        # state_dict = torch.load(config.model_base_path(p))
        # model = Model(config)
        # model.load_state_dict(state_dict)
        # model.to(device)
        # log param count
        log_to_file('Trainable params count', count_parameters(model))
        print(model.parameters())
        # exit()
        # OPTIMIZER
        optimizer = optim.SGD(model.parameters(), lr=config.start_lr)
        log_to_file("Optimizer", "SGD")
        # call backs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=config.loss_delta, patience=4,
                                               cooldown=4, verbose=True, min_lr=config.min_lr, factor=0.2)
        model_check_callback = ModelCheckPointCallBack(
           model,
           config.model_save_path(p*config.model_count+i),
           period=1,
           delta=config.loss_delta,
        )
        early_stop_callback = EarlyStopCallBack(patience=25, delta=config.loss_delta)

        # some vars
        epoch_loss = 0
        validation_loss = 0
        data_provider.new_epoch()
            # reset data provider
        # output the validation dataset

        # val_data_path = os.path.join(config.working_dir, 'val_data_{}.csv'.format(p*config.model_count+i))

        # val_df=pd.DataFrame(data_provider.validation_samples[i],columns=["hla_a","peptide", "ic50"])
        # val_df.to_csv(val_data_path, sep=',',header=True,index=True)
                
        steps = data_provider.train_steps()
        log_to_file('Start training1', datetime.datetime.now())
    

        for epoch in range(config.epochs):
            epoch_start_time = datetime.datetime.now()
            # train batches
            print(steps)
            model.train(True)
            for _ in range(steps):
                data = data_provider.batch_train(i)
                print("***")
                loss = batch_train(model, device, data, config)
                print("loss:",loss)
                # exit()
                loss.backward()
                # clip grads
                nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
                # update params
                optimizer.step()
                # record loss
                epoch_loss += loss.item()
                # reset grad
                optimizer.zero_grad()
                # time compute
            time_delta = datetime.datetime.now() - epoch_start_time
                # validation on epoch end

            model.eval()
            # print(data_provider.val_steps())
            # print(data_provider.batch_index_val)
            # validation_call
            val_sample=[]
            val_pred=[]
            for _ in range(data_provider.val_steps()):
                data = data_provider.batch_val(i)
                t_loss,t_pred,t_samples=batch_validation(model, device, data, config)
                val_sample.append(t_samples)
                val_pred.append(t_pred)
                validation_loss += t_loss
            # log
            log_to_file("Training process", "[base_model{0:1d}]-[model{1:1d}]-[Epoch {2:04d}] - time: {3:4d} s, train_loss: {4:0.5f}, val_loss: {5:0.5f}".format(
               p,i, epoch, time_delta.seconds, epoch_loss / steps, validation_loss / data_provider.val_steps()))
            # call back
            model_check_callback.check(epoch, validation_loss / data_provider.val_steps())
            if early_stop_callback.check(epoch, validation_loss / data_provider.val_steps()):
                break
            # LR schedule
            scheduler.step(loss.item())
            # reset loss
            epoch_loss = 0
            validation_loss = 0
            # reset data provider
            data_provider.new_epoch()
            # save last epoch model
            torch.save(model.state_dict(), os.path.join(config.working_dir, 'last_epoch_model_{}.pytorch'.format(p*config.model_count+i)))
        #validation_call
        val_path = os.path.join(config.working_dir, 'val_result_{}.csv'.format(p*config.model_count+i))

        val_temp_list=[]    
        for ii in range(len(val_sample)):
            for jj in range(len(val_sample[ii])):
                temp={"hla_a":val_sample[ii][jj][0],"peptide": val_sample[ii][jj][1], "ic50":val_sample[ii][jj][2], "pred_ic50": val_pred[ii][jj]}
                val_temp_list.append(temp)
        val_df=pd.DataFrame(val_temp_list)
        val_df["up_ic50"] = 50000**val_df["ic50"]
        val_df["up_pred_ic50"] = 50000**val_df["pred_ic50"]
        val_df["binding"] = val_df["up_ic50"].apply(lambda x:1 if x<500 else 0)
        val_df["pred_binding"]=val_df["up_pred_ic50"].apply(lambda x:1 if x<500 else 0)

        # val_df.to_csv(val_path, sep=',',header=True,index=True)
        # exit()


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

    log_to_file('Traning samples', len(data_provider[0].train_samples[0]))
    log_to_file('Val samples', len(data_provider[0].validation_samples[0]))
    log_to_file('Traning steps', data_provider[0].train_steps())
    log_to_file('Val steps', data_provider[0].val_steps())
    log_to_file('Batch size', data_provider[0].batch_size)
    log_to_file('max_len_hla', data_provider[0].max_len_hla)
    log_to_file('max_len_pep', data_provider[0].max_len_pep)
    
    for p in range(config.base_model_count):
        train(config, data_provider[p],p)


if __name__ == '__main__':
    main()
