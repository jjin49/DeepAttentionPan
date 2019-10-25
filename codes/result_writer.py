import os
import math
import numbers
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
#############################################################################################
#
# Write result
#
#############################################################################################

def weeekly_result_writer(result_dict, config):
    """Write prediction result as an additional column
    out [weekly_result.txt]
    """
    out_file_path = os.path.join(config.working_dir, 'weekly_result.txt')
    out_file = open(out_file_path, 'w')

    with open(config.test_file) as in_file:
        for line_num, line in enumerate(in_file):
            info = line.strip('\n').split('\t')
            if line_num == 0:
                # title
                out_str = '{}\t{}\t{}\n'.format(
                    '\t'.join(info[:7]),
                    'our_method_ic50',
                    '\t'.join(info[7:])
                )
                out_file.write(out_str)
            else:
                iedb_id = info[1]
                alleles = info[2]
                measure_type = info[4]
                peptide = info[5]

                hla_a = alleles

                uid = '{iedb_id}-{hla_a}-{peptide}-{measure_type}'.format(
                    iedb_id=iedb_id,
                    hla_a=hla_a,
                    # hla_b=hla_b,
                    peptide=peptide,
                    measure_type=measure_type,
                )

                if uid not in result_dict:
                    value = '-'
                else:
                    value = math.pow(50000,1-result_dict[uid])
                out_str = '{}\t{}\t{}\n'.format(
                    '\t'.join(info[:7]),
                    value,
                    '\t'.join(info[7:])
                )
                out_file.write(out_str)

    return out_file_path

#############################################################################################
#
# Write metrics
#
#############################################################################################

METHOD_LIST = [
    'our_method_ic50',
    'NetMHCpan 2.8',
    'NetMHCpan 3.0',
    'NetMHCpan 4.0',
    'SMM',
    'ANN 3.4',
    'ANN 4.0',
    'ARB',
    'SMMPMBEC',
    'IEDB Consensus',
    'NetMHCcons',
    'PickPocket',
]


def get_srcc(real, pred, measure_type):
    """
    """
    # all pred are ic50, neg them to get real correlation
    pred = [-x for x in pred]

    # if real also ic50, neg them
    if measure_type == 'ic50':
        real = [-x for x in real]

    return spearmanr(pred, real)[0]


def get_auc(real, pred, measure_type):
    """
    """
    # all pred are ic50, neg them to get real correlation
    pred = [-x for x in pred]

    # convert real to binary labels according to measure type
    real_binary = real
    if measure_type == 'ic50':
        real_binary = [1 if x < 500 else 0 for x in real]
    elif measure_type== 't1/2':
    	real_binary = [1 if x>120 else 0 for x in real]

    try:
        return roc_auc_score(real_binary, pred)
    except:
        return '-'


def write_metrics_file(result_file, config):
    """Reading [weekly_result.txt], write to [weekly_result_METRICS.txt]
    by each IEDB record
    """
    METRIC_PRECISION_DIGIT = 2

    out_file_path = os.path.join(config.working_dir, 'weekly_result_METRICS.txt')
    out_file = open(out_file_path, 'w')

    title = 'Date\tIEDB reference\tAllele\tPeptide length\tcount\tMeasurement type'
    for method_name in METHOD_LIST:
        title += '\t{method_name}_auc\t{method_name}_srcc'.format(method_name=method_name)
    out_file.write(title + '\n')
    
    # use to get max value for each record
    metric_max_info = {}
    for method_name in METHOD_LIST:
        metric_max_info[method_name] = [0, 0]

    result_info = get_weekly_result_info_dict(result_file)
    for record, info in result_info.items():
        date = info['date']
        iedb_id = info['iedb_id']
        pep_length = info['pep_length']
        measure_type = info['measure_type']
        allele = info['full_allele']
        label_values = info['label_values']
        count = len(label_values)
        
        # below only use when compare with ACME
        # if count<=5:
        # 	continue

        out_str = '{}\t{}\t{}\t{}\t{}\t{}'.format(
            date, iedb_id, allele, pep_length, count, measure_type
        )

        max_srcc = -1000000
        max_auc = -1000000
        srcc_list = []
        auc_list = []
        for method_name in METHOD_LIST:
            pred_vals = info['method_values'][method_name]
            if len(pred_vals) != count:
                srcc = '-'
                auc = '-'
            else:
                srcc = get_srcc(label_values, pred_vals, measure_type)
                auc  = get_auc(label_values, pred_vals, measure_type)

                if isinstance(srcc, numbers.Number):
                    srcc = round(float(srcc),2)
                    max_srcc = max(srcc, max_srcc)
                if isinstance(auc, numbers.Number):
                    auc  = round(float(auc),2)
                    max_auc = max(auc, max_auc)

            srcc_list.append(srcc)
            auc_list.append(auc)
            out_str += '\t{}\t{}'.format(auc, srcc)

        # update max auc, srcc count in this record
        for i, (srcc, auc) in enumerate(zip(srcc_list, auc_list)):
            if auc!='-' and auc == max_auc:
                metric_max_info[METHOD_LIST[i]][0] += 1
            if srcc!='-' and srcc== max_srcc:
                metric_max_info[METHOD_LIST[i]][1] += 1

        # write
        out_file.write(out_str + '\n')

    # write max win count
    out_str = '\t'.join(['-'] * 6)  # offset
    for method_name in METHOD_LIST:
        out_str += '\t{}\t{}'.format(metric_max_info[method_name][0], metric_max_info[method_name][1])
    out_file.write(out_str + '\n')

    return out_file_path


def get_weekly_result_info_dict(result_file):

    result_info = {}
    with open(result_file) as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue

            info = line.strip('\n').split('\t')
            date = info[0]
            iedb_id = info[1]
            full_allele = info[2]
            measure_type = info[4]
            pep_len = len(info[5])
            measure_value = float(info[6])

            record_id = '{}-{}-{}-{}'.format(iedb_id, full_allele, pep_len, measure_type)

            if record_id not in result_info:
                result_info[record_id] = {}
                result_info[record_id]['full_allele'] = full_allele
                result_info[record_id]['date'] = date
                result_info[record_id]['pep_length'] = pep_len
                result_info[record_id]['iedb_id'] = iedb_id
                result_info[record_id]['measure_type'] = measure_type
                result_info[record_id]['label_values'] = []
                result_info[record_id]['method_values'] = {}
                for method in METHOD_LIST:
                    result_info[record_id]['method_values'][method] = []

            # fill real value
            result_info[record_id]['label_values'].append(measure_value)

            # fill prediction values, if no result, do not fill
            for method_index, method_name in enumerate(METHOD_LIST):
                col_index = method_index + 7
                val = info[col_index]
                try:
                    val = float(val)
                    result_info[record_id]['method_values'][method_name].append(val)
                except:
                    pass

    return result_info


def main():
    pass

if __name__ == '__main__':
    main()
