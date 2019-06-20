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
LOSS = 'loss'
VAL_ACCURACY = 'val_acc'
VAL_LOSS = 'val_loss'
VP = 'vp'
FP = 'fp'
FN = 'fn'
F_MEASURE = 'f_measure'


def get_labels_filtered(labels):
    labels.sort()
    labels.reverse()
    return labels


def get_labels_from_confusion_matrix(matrix_dict, sep='-'):
    labels = set()
    for key in matrix_dict.keys():
        x, y = key.split(sep)
        labels.add(x)
        labels.add(y)
    return get_labels_filtered(list(labels))


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


def generate_folds(dataset, k_fold, test_folder):
    folds = []
    folds_indexes = []
    fold_test_size = int(len(dataset) / k_fold)
    indexes = dataset.index.tolist()

    random.shuffle(indexes)

    for x in range(k_fold - 1):
        test_indexes = indexes[x*fold_test_size:(x+1)*fold_test_size]
        folds_indexes.append(test_indexes)
        pass
    folds_indexes.append(indexes[(k_fold - 1) * fold_test_size:])

    for index, test_fold in enumerate(folds_indexes):
        selected_indexes = dataset.index.isin(test_fold)
        test_set = dataset.iloc[selected_indexes]
        training_set = dataset[~selected_indexes]

        fold_folder = '{}/fold-{}'.format(test_folder, index + 1)

        try:
            os.makedirs(fold_folder)
        except:
            pass

        train_file = '{}/train.txt'.format(fold_folder)
        test_file = '{}/test.txt'.format(fold_folder)

        parse_dataframe_to_txt(training_set, train_file)
        parse_dataframe_to_txt(test_set, test_file)

        # folds.append((training_set, test_set))
        folds.append((train_file, test_file))

    return folds


def get_f_measure(precision, recall, beta=1.0):
    if precision == 0 and recall == 0:
        return 0
    return (1 + math.pow(beta, 2)) * ((precision * recall)/((math.pow(beta, 2) * precision) + recall))


def get_empty_result_dict(labels):
    result_dict = {
        FOLD: [],
        EPOCH: [],
        ARCHITECTURE: [],
        LOSS: [],
        ACCURACY: [],
        VAL_ACCURACY: [],
        VAL_LOSS: [],
        PRECISION_MACRO: [],
        PRECISION_MICRO: [],
        RECALL_MACRO: [],
        RECALL_MICRO: [],
    }
    for label in labels:
        label_attr = '{}_{}'.format(LABELS, label)
        result_dict['{}_{}'.format(label_attr, VP)] = []
        result_dict['{}_{}'.format(label_attr, FP)] = []
        result_dict['{}_{}'.format(label_attr, FN)] = []
        result_dict['{}_{}'.format(label_attr, PRECISION)] = []
        result_dict['{}_{}'.format(label_attr, RECALL)] = []
        result_dict['{}_{}'.format(label_attr, F_MEASURE)] = []

    return result_dict


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


def describe_metrics(metrics):
    print()
    print('Accuracy: {}'.format(metrics[ACCURACY]))
    print('Error rate: {}'.format(1 - metrics[ACCURACY]))
    print('Precision Macro: {}'.format(metrics[PRECISION_MACRO]))
    print('Precision Micro: {}'.format(metrics[PRECISION_MICRO]))
    print('Recall Macro: {}'.format(metrics[RECALL_MACRO]))
    print('Recall Micro: {}'.format(metrics[RECALL_MICRO]))
    print()
    print('Confusion Matrix')
    print('-'*50)

    matrix = metrics[CONFUSION_MATRIX]

    for label in metrics[LABELS].keys():
        print('\t{}'.format(label), end='')
    print()
    for index, label in enumerate(metrics[LABELS].keys()):
        print('{}\t{}'.format(label, '\t'.join(
            [str(v) for v in matrix[index]])))

    for label in metrics[LABELS].keys():
        print('-'*50)
        print(label)

        vp = metrics[LABELS][label][VP]
        fp = metrics[LABELS][label][FP]
        fn = metrics[LABELS][label][FN]
        precision = metrics[LABELS][label][PRECISION]
        recall = metrics[LABELS][label][RECALL]
        f_measure = metrics[LABELS][label][F_MEASURE]

        print('  VP: {}'.format(vp))
        print('  FP: {}'.format(fp))
        print('  FN: {}'.format(fn))
        print('  Precision: {}'.format(precision))
        print('  Recall: {}'.format(recall))
        print('  F-Measure: {}'.format(f_measure))


def test_forest(test_data, forest):
    label_field = test_data.columns[-1]
    real_labels = test_data[label_field]

    unique_labels = real_labels.unique().tolist()

    predicted_labels = []
    for (index, (_, row)) in enumerate(test_data.iterrows()):
        predicted_labels.append([])
        for tree in forest:
            predicted_label = dt.classify_instance(tree, row)
            predicted_labels[index].append(predicted_label)
            if not predicted_label in unique_labels:
                unique_labels.append(predicted_label)

    logger.debug('PREDICTIONS')
    logger.debug('')
    logger.debug('\t'.join(['Tree {}'.format(i+1)
                            for i in range(len(forest))]))
    for p in predicted_labels:
        logger.debug('\t'.join(p))
    logger.debug('-'*50)
    logger.debug('')
    logger.debug('Forest Predicted\tReal')
    logger.debug('-'*30)

    matrix_dict = {}
    possible_labels = get_labels_filtered(unique_labels)
    for x_label in possible_labels:
        for y_label in possible_labels:
            matrix_dict['{}-{}'.format(x_label, y_label)] = 0

    for index, predictions in enumerate(predicted_labels):
        prediction = forest_decision(predictions)
        real_label = real_labels.values[index]
        if '{}-{}'.format(real_label, predicted_label) in matrix_dict:
            matrix_dict['{}-{}'.format(real_label, prediction)] += 1

        logger.debug('{}\t\t\t{}'.format(prediction, real_label))

    matrix = get_confusion_matrix_from_dict(matrix_dict)
    metrics = get_evaluation_metrics(matrix, possible_labels)

    return metrics


def cross_validation(dataset, k_fold, test_folder, original_labels, max_iterations, alpha,
                     beta=0.9, less_acceptable_difference=0.0001, momentum=True, patience=50, logger=logger):

    y_field = dataset.columns[-1]
    # possible_labels = dataset[y_field].unique().tolist()
    possible_labels = original_labels

    attributes = dataset.columns[:-1].tolist()

    result_dict = get_empty_result_dict(possible_labels)

    # generating folds

    folds = generate_folds(dataset, k_fold, test_folder)

    net_file = '{}/network.txt'.format(test_folder)
    weights_file = '{}/initial_weights.txt'.format(test_folder)

    for f_index, (train_file, test_file) in enumerate(folds):
        print('Processing fold {}'.format(f_index + 1))
        print('Training network')

        network, training_result = bp.backpropagation(net_file, weights_file, train_file,
                                                      max_iterations, alpha,
                                                      less_acceptable_difference=less_acceptable_difference,
                                                      validation_filename=test_file, possible_labels=possible_labels,
                                                      patience=patience, logger=logger)
        # forest = []


        # for t_index in range(tree_amount):
        #     print('Getting dataset bootstrap')
        #     bootstrap_train, _ = get_dataset_bootstrap(train_file)

        #     print('Generating tree {} with {} attributes'.format(
        #         t_index + 1, attributes_amount))
        #     tree = dt.generate_tree(
        #         bootstrap_train, attributes, logger=logger, m=attributes_amount)
        #     forest.append(tree)

        #     if img_folder:
        #         dt.export_dot(tree).to_picture(
        #             '{}/forest-{}-tree-{}.png'.format(img_folder.rstrip('/'), f_index + 1, t_index + 1))
        #     if json_folder:
        #         with open('{}/forest-{}-tree-{}.json'.format(json_folder.rstrip('/'), f_index + 1, t_index + 1), 'w') as tree_file:
        #             tree_file.write(json_exporter.export(tree))

        # print('Testing forest with {} instances'.format(len(test_file)))
        # metrics = test_forest(test_file, forest)
        # print('Testing finished')
        # print('')
        # add_metrics_to_dict(result_dict, metrics)
        epochs_trained = len(training_result[LOSS])

        training_result[FOLD].extend([f_index + 1] * epochs_trained)
        training_result[EPOCH].extend(list(range(1, epochs_trained + 1)))

        for k in result_dict.keys():
            result_dict[k].extend(training_result[k])

        # if logger.isEnabledFor(logging.INFO):
        #     describe_metrics(metrics)

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
@click.option('--initial-layer', '-il', default=1)
@click.option('--initial-neuron', '-in', default=1)
@click.option('--last-layer', '-ll', default=1)
@click.option('--last-neuron', '-ln', default=1)
@click.option('--alpha', '-a', default=0.1, help='Weights Update Rate, is used to smooth the gradient')
@click.option('--beta', '-b', default=0.9, help='Relevance of recent average direction (Method of Moment)')
@click.option('--regularization', '-r', default=0.25)
@click.option('--max-iterations', '-m', default=10)
@click.option('--patience', '-p', default=50)
def execute(filename, separator, k_fold, initial_layer,
            initial_neuron, last_layer, last_neuron,
            alpha, beta, regularization, max_iterations, patience):
    """Execute a neural network cross validation"""

    now = datetime.datetime.now()

    test_folder = 'inputs/tests/{}'.format(now.strftime('%Y-%m-%d_%Hh%M'))

    try:
        os.makedirs(test_folder)
    except:
        pass

    dataset = pd.read_csv(filename, sep=separator)
    original_labels = dataset[dataset.columns[-1]].unique().tolist()
    attributes = dataset.columns[:-1].tolist()
    # dataset[y_field] = dataset[y_field].astype(str)

    # print(dataset.sample(5))

    # Normalize dataset
    dataset = normalize_dataframe(dataset)

    # Parsing dataset
    parse_dataframe_to_txt(dataset, '{}/dataset.txt'.format(test_folder))
    # y_field = dataset.columns[-1]
    # labels = dataset[y_field].unique().tolist()
    # attributes = dataset.columns[:-1].tolist()

    # data_txt = ''

    # for _, row in dataset.iterrows():
    #     x_attr = []
    #     for attr in attributes:
    #         x_attr.append(str(row[attr]))
    #     y_attr = len(labels) * ['0.0']
    #     y_attr[labels.index(row[y_field])] = '1.0'

    #     data_txt += '{}; {}\n'.format(', '.join(x_attr), ', '.join(y_attr))

    # with open('{}/dataset.txt'.format(test_folder), 'w') as dataset_file:
    #     dataset_file.write(data_txt)

    test = 1

    test_results = get_empty_result_dict(original_labels)

    start = time.time()
    for layer_amount in range(initial_layer, last_layer + 1):
        for neuron_amount in range(initial_neuron, last_neuron + 1):
            net_arch = [len(attributes)] + layer_amount * \
                [neuron_amount] + [len(original_labels)]

            print(layer_amount, neuron_amount, net_arch)

            net_folder = '{}/r-{}-architecture-{}'.format(
                test_folder, regularization, '-'.join((str(x) for x in net_arch)))

            try:
                os.makedirs(net_folder)
            except:
                pass

            # Generating weights
            weights = []
            for i, n in enumerate(net_arch[:-1]):
                l_weights = []
                for x in range(net_arch[i+1]):
                    # l_weights.append(', '.join(['{:.2f}'.format(random.uniform(-1, 1))
                    #                             for x in range(n + 1)]))
                    l_weights.append(
                        ', '.join(['{:.2f}'.format(x) for x in np.random.normal(size=n+1)]))
                # print(l_weights)
                weights.append('; '.join(l_weights))

            weights_txt = '\n'.join(weights)
            # print('-'*50)
            # print('WEIGHTS')
            # print('-'*50)
            # print(weights_txt)
            with open('{}/initial_weights.txt'.format(net_folder), 'w') as w_file:
                w_file.write(weights_txt)

            # Generating Architecture
            arch_txt = '\n'.join(str(v) for v in [regularization] + net_arch)

            # print('-'*50)
            # print('ARCHITECTURE')
            # print('-'*50)
            # print(arch_txt)
            with open('{}/network.txt'.format(net_folder), 'w') as a_file:
                a_file.write(arch_txt)

            # training nn with:
            # architecture, weights, alpha, j_threshold, max_iterations,
            # batch_size, regularization_factor

            # -----------
            # UNIQUE TRAINING HAS TO REMOVE COMMENTS
            # network, training_result = bp.backpropagation('{}/network.txt'.format(net_folder),
            #                                                 '{}/initial_weights.txt'.format(net_folder),
            #                                                 '{}/dataset.txt'.format(test_folder),
            #                                                 max_iterations, alpha, logger=logger,
            #                                                 less_acceptable_difference=0.000001)

            # -----------

                     # dataset, k_fold, test_folder, max_iterations, alpha, beta=0.9, less_acceptable_difference=0.0001,
#                      momentum=True, logger=logger
            r = cross_validation(dataset, k_fold, net_folder, original_labels, max_iterations, alpha,
                                 beta=beta, less_acceptable_difference=0.000001, 
                                 patience=patience, logger=logger)

            epochs_trained = len(r[LOSS])

            # r[FOLD].extend([test] * epochs_trained)
            # r[EPOCH].extend(list(range(1, epochs_trained + 1)))
            r[ARCHITECTURE].extend([net_arch] * epochs_trained)

            # print('TEST Results')
            # print(test_results)
            # # print('\n\n\nDATA Test')
            # # print(pd.DataFrame(test_results))
            # print('\n\n\nRRR')
            # print(r)
            # print('\\\\DATA R')
            # print(pd.DataFrame(r))
            for k in r.keys():
                test_results[k].extend(r[k])


            # print(test_results)

            result_frame = pd.DataFrame(test_results)

            # print(result_frame.head(5))
            # print(result_frame.tail(5))
            
            result_frame.to_csv('{}/result.csv'.format(test_folder), index=False)

            # if test_result_output:
            #     result_frame = pd.DataFrame(test_results)
            #     result_frame.to_csv(test_result_output, index=False)
            # getting result

            # -------------

            # result csv
            # test epoch arch loss acc val_acc val_loss
            # prec_macro, preci_micro, rec_macro, rec_micro,
            # labels_M_vp,labels_M_fp,labels_M_fn,labels_M_precision,labels_M_recall,labels_M_f_measure

            # epochs_trained = len(training_result['loss'])

            # result_final['test'].extend([test] * epochs_trained)
            # result_final['epoch'].extend(list(range(1, epochs_trained + 1)))
            # result_final['architecture'].extend([net_arch] * epochs_trained)
            # result_final['loss'].extend(training_result['loss'])
            # result_final['acc'].extend(training_result['acc'])

            test += 1
            print()

    end = time.time()

    print('Cross validation finished')
    print()
    print('Time elapsed {} seconds'.format(end - start))

    # print(test_results)

    result_frame = pd.DataFrame(test_results)
    # print(result_frame.head(5))
    # print(result_frame.tail(5))
    result_frame.to_csv('{}/result.csv'.format(test_folder), index=False)


if __name__ == "__main__":
    main()
