import os
import json
import csv
import numpy as np

PATH_DATASET = '/home/zeyuzhang/PycharmProjects/acl_dataprocess/data'
PATH_PREDICTIONS = '/home/zeyuzhang/PycharmProjects/acl_dataprocess/predictions'
PATH_OUTPUT = '/home/zeyuzhang/PycharmProjects/acl_dataprocess'
def read_top_predictions(data_path, dataset_name, data_fold, data_type, top_number):
    predictions = []
    if data_type == 'dev':
        data_type = 'eval'
    predictions_path = os.path.join(data_path, dataset_name, data_fold+'_predictions', data_type+'_predictions.txt')
    with open(predictions_path, 'r+') as file:
        for line in file.readlines():
            scores_temp = line.strip().split(' ')
            scores = list(map(float, scores_temp))
            labels_index = np.argsort(scores)[-int(top_number):]
            predictions.append(list(labels_index))
    return predictions

def read_original_data(data_path, dataset_name, data_fold, data_type):
    original_data = []
    original_data_path = os.path.join(data_path, dataset_name, data_fold, data_type+'.tsv')
    with open(original_data_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            original_data.append(row)
    return original_data

def read_labels(data_path, dataset_name):
    labels_path = os.path.join(data_path, dataset_name,'label.txt')
    with open(labels_path, 'r') as file:
        for line in file.readlines():
            splits = line.strip()[1:-1].split(', ')
            labels = list(map(lambda x: x[1:-1],splits))
    return labels

def create_output(top_number, dataset_name, data_fold, data_flag, data_type):
    final_data = []
    original_data = read_original_data(PATH_DATASET, dataset_name, data_fold, data_type)
    predictions = read_top_predictions(PATH_PREDICTIONS, dataset_name, data_fold, data_type, top_number)
    labels = read_labels(PATH_DATASET, dataset_name)
    if data_flag == 'real':
        for idx, mentions in enumerate(original_data):
            single_data = []
            single_data.append(mentions[0])
            predicted_labels = [labels[item] for item in predictions[idx]]
            single_data.extend(predicted_labels)
            if mentions[1] in predicted_labels:
                single_data.append(str(predicted_labels.index(mentions[1])))
            else:
                single_data.append('-1')
            final_data.append(single_data)
    else:
        for idx, mentions in enumerate(original_data):
            single_data = []
            single_data.append(mentions[0])
            predicted_labels = [labels[item] for item in predictions[idx]]
            if mentions[1] in predicted_labels:
                single_data.extend(predicted_labels)
                single_data.append(str(predicted_labels.index(mentions[1])))
            else:
                final_predicted_labels = predicted_labels[1:].copy()
                final_predicted_labels.append(mentions[1])
                single_data.extend(final_predicted_labels)
                single_data.append(str(final_predicted_labels.index(mentions[1])))
            final_data.append(single_data)
    print('s{} {} {} {} {} is completede.'.format(top_number, dataset_name, data_fold, data_flag, data_type))
    return final_data

def save_final_tsv(output_path, data_type, output_data):
    with open(os.path.join(output_path,data_type+'.tsv'), 'w') as output_file:
        tsv_writer = csv.writer(output_file, delimiter='\t')
        for data_item in output_data:
            tsv_writer.writerow(data_item)

def main():
    for top_number in ['10','20','30','40']:
        output_top_number_path = os.path.join(PATH_OUTPUT,'s'+top_number)
        if not os.path.exists(output_top_number_path):
            os.makedirs(output_top_number_path)
        for dataset_name in ['AskAPatient', 'TwADR-L']:
            output_dataset_path = os.path.join(output_top_number_path, dataset_name)
            if not os.path.exists(output_dataset_path):
                os.makedirs(output_dataset_path)
            for data_fold in [str(i) for i in range(10)]:
                output_data_fold_path = os.path.join(output_dataset_path, data_fold)
                if not os.path.exists(output_data_fold_path):
                    os.makedirs(output_data_fold_path)
                for data_flag in ['real', 'oracle']:
                    output_data_flag_path = os.path.join(output_data_fold_path, data_flag)
                    if not os.path.exists(output_data_flag_path):
                        os.makedirs(output_data_flag_path)
                    for data_type in ['train', 'dev', 'test']:
                        output_data = create_output(top_number, dataset_name, data_fold, data_flag, data_type)
                        save_final_tsv(output_data_flag_path, data_type, output_data)

if __name__ == '__main__':

    main()

