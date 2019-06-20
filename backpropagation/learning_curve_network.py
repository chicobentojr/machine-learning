import os
import logging
import math
import random
import time
import datetime
import click
import click_log
import json
import pandas as pd
import numpy as np
import matrix as mt
import network_classes as net
import backpropagation as bp
import cross_validation_network as cvn
from sklearn import preprocessing

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

ACCURACY = 'acc'
ERROR_RATE = 'error_rate'
RECALL = 'recall'
PRECISION = 'precision'
LABELS = 'labels'
PRECISION_MACRO = 'precision_macro'
PRECISION_MICRO = 'precision_micro'
RECALL_MACRO = 'recall_macro'
RECALL_MICRO = 'recall_micro'
CONFUSION_MATRIX = 'confusion_matrix'
FOLD = 'fold'
EPOCH = 'epoch'
ARCHITECTURE = 'architecture'
REGULARIZATION = 'regularization'
LOSS = 'loss'
VAL_ACCURACY = 'val_acc'
VAL_LOSS = 'val_loss'
VP = 'vp'
FP = 'fp'
FN = 'fn'
F_MEASURE = 'f_measure'
TRAINING_EXAMPLES = 'examples'


def get_labels_from_confusion_matrix(matrix_dict, sep='-'):
    labels = set()
    for key in matrix_dict.keys():
        x, y = key.split(sep)
        labels.add(x)
        labels.add(y)
    return list(labels)


def get_confusion_matrix_from_dict(matrix_dict, sep='-'):
    labels = get_labels_from_confusion_matrix(matrix_dict, sep)
    m = []
    for index, x_label in enumerate(labels):
        m.append([])
        for y_label in labels:
            m[index].append(
                matrix_dict['{}{}{}'.format(x_label, sep, y_label)])
    return m


def get_evaluation_metrics(matrix, labels):
    metrics = {
        ACCURACY: 0,
        PRECISION_MACRO: 0,
        PRECISION_MICRO: 0,
        RECALL_MACRO: 0,
        RECALL_MICRO: 0,
        LABELS: {},
        CONFUSION_MATRIX: matrix
    }
    n = 0
    total_vp = 0
    total_fp = 0
    total_fn = 0

    for x_i in range(len(labels)):
        metrics[ACCURACY] += matrix[x_i][x_i]
        metrics[LABELS][labels[x_i]] = {VP: 0, FP: 0, FN: 0}
        for y_i in range(len(labels)):
            n += matrix[x_i][y_i]
            if x_i == y_i:
                metrics[LABELS][labels[x_i]][VP] = matrix[x_i][y_i]
            else:
                metrics[LABELS][labels[x_i]][FN] += matrix[x_i][y_i]
                metrics[LABELS][labels[x_i]][FP] += matrix[y_i][x_i]

    for x in labels:
        vp = metrics[LABELS][x][VP]
        fn = metrics[LABELS][x][FN]
        fp = metrics[LABELS][x][FP]

        total_vp += vp
        total_fp += fp
        total_fn += fn

        precision = vp / (vp + fp) if vp > 0 else 0
        recall = vp / (vp + fn) if vp > 0 else 0

        metrics[LABELS][x][PRECISION] = precision
        metrics[LABELS][x][RECALL] = recall
        metrics[LABELS][x][F_MEASURE] = get_f_measure(precision, recall)

        metrics[PRECISION_MACRO] += metrics[LABELS][x][PRECISION]
        metrics[RECALL_MACRO] += metrics[LABELS][x][RECALL]

    metrics[ACCURACY] = metrics[ACCURACY] / n
    metrics[ERROR_RATE] = 1 - metrics[ACCURACY]
    metrics[PRECISION_MACRO] = metrics[PRECISION_MACRO] / len(labels)
    metrics[PRECISION_MICRO] = total_vp / (total_vp + total_fp)
    metrics[RECALL_MACRO] = metrics[RECALL_MACRO] / len(labels)
    metrics[RECALL_MICRO] = total_vp / (total_vp + total_fn)

    return metrics


def get_dataset_bootstrap(d):
    training = pd.DataFrame()
    test = None
    indexes = []

    for _ in range(len(d)):
        new_row = random.randint(0, len(d) - 1)
        indexes.append(new_row)
        training = training.append(d.iloc[new_row])

    selected_rows = d.index.isin(indexes)

    training = training.reindex(d.columns, axis=1)
    test = d[~selected_rows]

    return (training, test)


def generate_data_steps(dataset, step, test_folder):
    steps = []
    test_size = len(dataset) // 10
    indexes = dataset.index.tolist()

    random.shuffle(indexes)

    test_indexes = indexes[:test_size]
    training_indexes = indexes[test_size:]

    selected_indexes = dataset.index.isin(test_indexes)
    test_set = dataset.iloc[selected_indexes]
    
    for x in range(0, len(training_indexes), step):

        step_indexes = dataset.index.isin(training_indexes[:x+step])
        training_set = dataset[step_indexes]

        fold_folder = '{}/step-{}'.format(test_folder, x+step)
        try:
            os.makedirs(fold_folder)
        except:
            pass

        train_file = '{}/train.txt'.format(fold_folder)
        test_file = '{}/test.txt'.format(fold_folder)

        parse_dataframe_to_txt(training_set, train_file)
        parse_dataframe_to_txt(test_set, test_file)

        steps.append((train_file, test_file))
    return steps


def get_f_measure(precision, recall, beta=1.0):
    if precision == 0 and recall == 0:
        return 0
    return (1 + math.pow(beta, 2)) * ((precision * recall)/((math.pow(beta, 2) * precision) + recall))



def add_metrics_to_dict(result_dict, metrics):
    result_dict[PRECISION_MACRO].append(metrics[PRECISION_MACRO])
    result_dict[PRECISION_MICRO].append(metrics[PRECISION_MICRO])
    result_dict[RECALL_MACRO].append(metrics[RECALL_MACRO])
    result_dict[RECALL_MICRO].append(metrics[RECALL_MICRO])

    for label in metrics[LABELS].keys():
        label_attr = '{}_{}'.format(LABELS, label)

        result_dict['{}_{}'.format(label_attr, VP)].append(
            metrics[LABELS][label][VP])
        result_dict['{}_{}'.format(label_attr, FP)].append(
            metrics[LABELS][label][FP])
        result_dict['{}_{}'.format(label_attr, FN)].append(
            metrics[LABELS][label][FN])
        result_dict['{}_{}'.format(label_attr, PRECISION)].append(
            metrics[LABELS][label][PRECISION])
        result_dict['{}_{}'.format(label_attr, RECALL)].append(
            metrics[LABELS][label][RECALL])
        result_dict['{}_{}'.format(label_attr, F_MEASURE)].append(
            metrics[LABELS][label][F_MEASURE])

    return result_dict

def step_training(dataset, step, test_folder, original_labels, alpha,
                  beta=0.9, momentum=True, logger=logger):

    result_dict = {
        ARCHITECTURE: [],
        REGULARIZATION: [],
        TRAINING_EXAMPLES: [],
        LOSS: [],
        ACCURACY: [],
        VAL_ACCURACY: [],
        VAL_LOSS: [],
    }

    folds = generate_data_steps(dataset, step, test_folder)

    net_file = '{}/network.txt'.format(test_folder)
    weights_file = '{}/initial_weights.txt'.format(test_folder)

    for f_index, (train_file, test_file) in enumerate(folds):
        print('Processing fold {}'.format(f_index + 1))
        print('Training network')

        network, training_result = bp.backpropagation(net_file, weights_file, train_file,
                                                      1, alpha, validation_filename=test_file, 
                                                      possible_labels=original_labels, logger=logger)
        epochs_trained = len(training_result[LOSS])


        net_arch = [l.size for l in network.layers]

        training_result[ARCHITECTURE] = [net_arch] * epochs_trained
        training_result[REGULARIZATION] = [network.regularizationFactor] * epochs_trained
        training_result[TRAINING_EXAMPLES] = [f_index * step + step]

        for k in result_dict.keys():
            result_dict[k].extend(training_result[k])

        pd.DataFrame(result_dict).to_csv('{}/result.csv'.format(test_folder), index=False)

    return result_dict


def normalize_dataframe(dataset):
    x = dataset.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)


def parse_dataframe_to_txt(dataframe, output_file):
    y_field = dataframe.columns[-1]
    labels = dataframe[y_field].unique().tolist()
    attributes = dataframe.columns[:-1].tolist()

    data_txt = ''

    for _, row in dataframe.iterrows():
        x_attr = []
        for attr in attributes:
            x_attr.append(str(row[attr]))
        y_attr = len(labels) * ['0.0']
        y_attr[labels.index(row[y_field])] = '1.0'

        data_txt += '{}; {}\n'.format(', '.join(x_attr), ', '.join(y_attr))

    with open(output_file, 'w') as dataframe_file:
        dataframe_file.write(data_txt)


@click.group()
def main():
    pass


@main.command(name='execute')
@click_log.simple_verbosity_option(logger)
@click.argument('filename')
@click.option('--separator', '-s', default=',', help='your custom CSV separator (e.g.: ; or :)')
@click.option('--k-fold', '-k', default=5, help='your number of folds to cross validation')
@click.option('--alpha', '-a', default=0.1, help='Weights Update Rate, is used to smooth the gradient')
@click.option('--beta', '-b', default=0.9, help='Relevance of recent average direction (Method of Moment)')
@click.option('--regularizations', '-r', multiple=True, type=float)
@click.option('--max-iterations', '-m', default=10)
@click.option('--patience', '-p', default=50)
@click.option('--architectures', '-arc', multiple=True)
@click.option('--momentum/--no-momentum', default=True)
@click.option('--train-step', '-ts', default=5)
def execute(filename, separator, k_fold, alpha, beta, regularizations, max_iterations,
            patience, architectures, momentum, train_step):
    """Execute multiple neural networks cross validations"""

    data_file = filename.split('/')[-1].split('.')[0]

    now = datetime.datetime.now()

    test_folder = 'inputs/learning/{}_{}'.format(now.strftime('%Y-%m-%d_%Hh%M'), data_file)

    try:
        os.makedirs(test_folder)
    except:
        pass

    dataset = pd.read_csv(filename, sep=separator)
    original_labels = dataset[dataset.columns[-1]].unique().tolist()
    attributes = dataset.columns[:-1].tolist()

    # Normalize dataset
    dataset = normalize_dataframe(dataset)

    # Parsing dataset
    parse_dataframe_to_txt(dataset, '{}/dataset.txt'.format(test_folder))

    test = 1

    test_results = {
        ARCHITECTURE: [],
        REGULARIZATION: [],
        TRAINING_EXAMPLES: [],
        LOSS: [],
        ACCURACY: [],
        VAL_ACCURACY: [],
        VAL_LOSS: [],
    }

    start = time.time()
    for arch in architectures:
        for reg in regularizations:
            net_arch = [len(attributes)] + [int(l) for l in arch.split(',')]  + [len(original_labels)]

            logger.info('Arch = {}'.format(arch))
            logger.info('Regularization = {}'.format(reg))

            net_folder = '{}/r-{}-architecture-{}'.format(test_folder, reg, '-'.join((str(x) for x in net_arch)))

            try:
                os.makedirs(net_folder)
            except:
                pass

            # Generating weights
            weights = []
            for i, n in enumerate(net_arch[:-1]):
                l_weights = []
                for _ in range(net_arch[i+1]):
                    l_weights.append(
                        ', '.join(['{:.2f}'.format(x) for x in np.random.normal(size=n+1)]))

                weights.append('; '.join(l_weights))

            weights_txt = '\n'.join(weights)
            with open('{}/initial_weights.txt'.format(net_folder), 'w') as w_file:
                w_file.write(weights_txt)

            # Generating Architecture
            arch_txt = '\n'.join(str(v) for v in [reg] + net_arch)

            with open('{}/network.txt'.format(net_folder), 'w') as a_file:
                a_file.write(arch_txt)

            r = step_training(dataset, train_step, net_folder, original_labels, alpha,
                                 beta=beta, momentum=momentum, logger=logger)
            for k in r.keys():
                test_results[k].extend(r[k])

            result_frame = pd.DataFrame(test_results)
            result_frame.to_csv('{}/final-result.csv'.format(test_folder), index=False)

            test += 1
            print()

    end = time.time()

    print('Cross validation finished')
    print()
    print('Time elapsed {} seconds'.format(end - start))

    result_frame = pd.DataFrame(test_results)    
    result_frame.to_csv('{}/final-result.csv'.format(test_folder), index=False)


if __name__ == "__main__":
    main()
