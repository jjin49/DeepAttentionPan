import os
import math
import random

import torch

############################################################################
# Data provider
############################################################################


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class DataProvider:
    def __init__(self, sequence_encode_func, sequence_encode_func2, data_file, test_file, batch_size, max_len_hla=273, max_len_pep=37,
      model_count=5, shuffle=True):
        self.batch_size = batch_size
        self.data_file = data_file

        self.test_file = test_file
        self.sequence_encode_func = sequence_encode_func
        self.sequence_encode_func2= sequence_encode_func2
        self.shuffle = shuffle
        self.max_len_hla = max_len_hla
        self.max_len_pep = max_len_pep
        self.model_count=model_count

        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0
        # cache
        self.pep_encode_dict = {}
        self.hla_encode_dict = {}
        self.hla_encode_dict2= {}
        self.pep_encode_dict2= {}

        self.hla_sequence = {}
        self.read_hla_sequences()

        self.samples = []  
        self.train_samples = []
        self.validation_samples = []
        self.read_training_data()
        self.split_train_and_val()

        self.weekly_samples = []
        self.read_weekly_data()


    def train_steps(self):
        return math.ceil(len(self.train_samples[0]) / self.batch_size)

    def val_steps(self):
        return math.ceil(len(self.validation_samples[0]) / self.batch_size)

    def test_steps(self):
        return math.ceil(len(self.weekly_samples) / self.batch_size)

    def read_hla_sequences(self):

        file_path = os.path.join(BASE_DIR, '..','dataset',  'mhc_i_protein_seq2.txt')
        with open(file_path, 'r') as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split(' ')
                seq=info[1]
                if(len(seq)>=self.max_len_hla):
                   seq=seq[:self.max_len_hla]
                # print(info)
                self.hla_sequence[info[0]] = seq


    def read_weekly_data(self):

        with open(self.test_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                iedb_id = info[1]
                alleles = info[2]
                measure_type = info[4]
                peptide = info[5]
                org_pep=peptide[:]
                if len(peptide) > self.max_len_pep:
                    continue
                hla_a = alleles
                if hla_a not in self.hla_sequence :
                    continue
                uid = '{iedb_id}-{hla_a}-{peptide}-{measure_type}'.format(
                    iedb_id=iedb_id,
                    hla_a=hla_a,
                    peptide=org_pep,
                    measure_type=measure_type
                )
                # print(uid)
                self.weekly_samples.append((hla_a,peptide, uid))
            
    def read_training_data(self):

        with open(self.data_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                # print(info)

                hla_a = info[1]

                if hla_a not in self.hla_sequence :
                    continue

                peptide = info[3]
                if len(peptide) > self.max_len_pep:
                    continue

                ic50 = float(info[-1])
                ic50=1-math.log(ic50,50000)

                self.samples.append((hla_a,peptide, ic50))

        if self.shuffle:
            random.shuffle(self.samples)

        # exit()

    def split_train_and_val(self):

        vd_count=math.ceil(len(self.samples)/max(self.model_count,5))
        for i in range(max(self.model_count-1,4)):
            self.validation_samples.append(self.samples[i*vd_count:(i+1)*vd_count])
            temp_sample=self.samples[:]
            del(temp_sample[i*vd_count:(i+1)*vd_count])
            self.train_samples.append(temp_sample)


        self.validation_samples.append(self.samples[len(self.samples)-vd_count:])
        temp_sample=self.samples[:]
        del(temp_sample[len(self.samples)-vd_count:])
        self.train_samples.append(temp_sample)


    def batch_train(self,order):
        """A batch of training data
        """
        data = self.batch(self.batch_index_train, self.train_samples[order])
        self.batch_index_train += 1
        return data

    def batch_val(self,order):
        """A batch of validation data
        """
        data = self.batch(self.batch_index_val, self.validation_samples[order])
        self.batch_index_val += 1
        return data

    def batch_test(self):
        """A batch of test data
        """
        data = self.batch(self.batch_index_test, self.weekly_samples, testing=True)
        self.batch_index_test += 1
        return data

    def new_epoch(self):
        """New epoch. Reset batch index
        """
        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0

    def batch(self, batch_index, sample_set, testing=False):
        """Get a batch of samples
        """
        hla_a_tensors = []
        hla_a_mask = []
        hla_a_tensors2 = []
        hla_a_mask2 = []        
        pep_tensors = []
        pep_mask = []
        pep_tensors2 = []
        pep_mask2 = []
        ic50_list = []

        # for testing
        uid_list = []
        #validation_call
        sample_prototype=[]


        def encode_sample(sample):
            hla_a_allele = sample[0]
            # hla_b_allele = sample[1]
            pep = sample[1]

            if not testing:
                ic50 = sample[2]
            else:
                uid = sample[2]

            if hla_a_allele not in self.hla_encode_dict:
                hla_a_tensor, mask = self.sequence_encode_func(self.hla_sequence[hla_a_allele], self.max_len_hla)
                self.hla_encode_dict[hla_a_allele] = (hla_a_tensor, mask)

            hla_a_tensors.append(self.hla_encode_dict[hla_a_allele][0])
            hla_a_mask.append(self.hla_encode_dict[hla_a_allele][1])

            if hla_a_allele not in self.hla_encode_dict2:
                hla_a_tensor2, mask2 = self.sequence_encode_func2(self.hla_sequence[hla_a_allele], self.max_len_hla)
                self.hla_encode_dict2[hla_a_allele] = (hla_a_tensor2, mask2)

            hla_a_tensors2.append(self.hla_encode_dict2[hla_a_allele][0])
            hla_a_mask2.append(self.hla_encode_dict2[hla_a_allele][1])

            if pep not in self.pep_encode_dict:
                pep_tensor, mask = self.sequence_encode_func(pep, self.max_len_pep)              
                self.pep_encode_dict[pep] = (pep_tensor, mask)
            pep_tensors.append(self.pep_encode_dict[pep][0])
            pep_mask.append(self.pep_encode_dict[pep][1])

            if pep not in self.pep_encode_dict2:
                pep_tensor2, mask2 = self.sequence_encode_func2(pep, self.max_len_pep)
                self.pep_encode_dict2[pep] = (pep_tensor2, mask2)
            pep_tensors2.append(self.pep_encode_dict2[pep][0])
            pep_mask2.append(self.pep_encode_dict2[pep][1])


            if not testing:
                ic50_list.append(ic50)
            else:
                uid_list.append(uid)

        start_i = batch_index * self.batch_size
        end_i = start_i + self.batch_size
        for sample in sample_set[start_i: end_i]:
           # doesn't matter if the end_i exceed the maximum index
           #validation_call
           sample_prototype.append(sample)
           encode_sample(sample)
           # print(self.hla_encode_dict)
           # exit()
        

        # print("xxx")
        # in case last batch does not have enough samples, random get from previous samples
        if len(hla_a_tensors) < self.batch_size:
            if len(sample_set) < self.batch_size:
                for _ in range(self.batch_size - len(hla_a_tensors)):
                    #validation_call
                    temp=random.choice(sample_set)
                    sample_prototype.append(temp)                    
                    encode_sample(temp)

            else:
                for i in random.sample(range(start_i), self.batch_size - len(hla_a_tensors)):
                    #validation_call
                    sample_prototype.append(sample_set[i])
                    encode_sample(sample_set[i])

        if not testing:
     	
            return (
                torch.stack(hla_a_tensors, dim=0),
                torch.stack(hla_a_mask, dim=0),
                torch.stack(hla_a_tensors2, dim=0),
                torch.stack(hla_a_mask2, dim=0),


                torch.stack(pep_tensors, dim=0),
                torch.stack(pep_mask, dim=0),
                torch.stack(pep_tensors2, dim=0),
                torch.stack(pep_mask2, dim=0),

                torch.tensor(ic50_list),
                #validation_call
                sample_prototype  

            )
        else:
            return (
                torch.stack(hla_a_tensors, dim=0),
                torch.stack(hla_a_mask, dim=0),
                torch.stack(hla_a_tensors2, dim=0),
                torch.stack(hla_a_mask2, dim=0),

                torch.stack(pep_tensors, dim=0),
                torch.stack(pep_mask, dim=0),
                torch.stack(pep_tensors2, dim=0),
                torch.stack(pep_mask2, dim=0),

                uid_list,
            )


