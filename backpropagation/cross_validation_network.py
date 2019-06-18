import os
import logging
import math
import random
import time
import click
import click_log
import json
import pandas as pd

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

ACCURACY = 'accuracy'
ERROR_RATE = 'error_rate'
RECALL = 'recall'
PRECISION = 'precision'
LABELS = 'labels'
PRECISION_MACRO = 'precision_macro'
PRECISION_MICRO = 'precision_micro'
RECALL_MACRO = 'recall_macro'
RECALL_MICRO = 'recall_micro'
TREE_AMOUNT = 'tree_amount'
CONFUSION_MATRIX = 'confusion_matrix'
ATTRIBUTE_AMOUNT = 'attribute_amount'
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


def generate_folds(dataset, k_fold):
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

    for test_fold in folds_indexes:
        selected_indexes = dataset.index.isin(test_fold)
        test_set = dataset.iloc[selected_indexes]
        training_set = dataset[~selected_indexes]
        folds.append((training_set, test_set))

    return folds


def get_f_measure(precision, recall, beta=1.0):
    if precision == 0 and recall == 0:
        return 0
    return (1 + math.pow(beta, 2)) * ((precision * recall)/((math.pow(beta, 2) * precision) + recall))


def get_empty_result_dict(labels):
    result_dict = {
        TREE_AMOUNT: [],
        ATTRIBUTE_AMOUNT: [],
        ACCURACY: [],
        ERROR_RATE: [],
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


def add_metrics_to_dict(result_dict, metrics, tree_amount, attributes_amount):
    result_dict[ACCURACY].append(metrics[ACCURACY])
    result_dict[ERROR_RATE].append(metrics[ERROR_RATE])
    result_dict[PRECISION_MACRO].append(metrics[PRECISION_MACRO])
    result_dict[PRECISION_MICRO].append(metrics[PRECISION_MICRO])
    result_dict[RECALL_MACRO].append(metrics[RECALL_MACRO])
    result_dict[RECALL_MICRO].append(metrics[RECALL_MICRO])
    result_dict[TREE_AMOUNT].append(tree_amount)
    result_dict[ATTRIBUTE_AMOUNT].append(attributes_amount)

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


def cross_validation(dataset, k_fold, tree_amount, attributes_amount, img_folder, json_folder):
    attributes = dataset.columns[:-1].tolist()
    y_field = dataset.columns[-1]
    folds = generate_folds(dataset, k_fold)
    possible_labels = dataset[y_field].unique().tolist()

    result_dict = get_empty_result_dict(possible_labels)

    for f_index, (train, test) in enumerate(folds):
        logger.info('Processing fold {}'.format(f_index + 1))
        logger.info('Training {} trees with {} instances'.format(
            tree_amount, len(train)))

        forest = []

        for t_index in range(tree_amount):
            logger.info('Getting dataset bootstrap')
            bootstrap_train, _ = get_dataset_bootstrap(train)

            logger.info('Generating tree {} with {} attributes'.format(
                t_index + 1, attributes_amount))
            tree = dt.generate_tree(
                bootstrap_train, attributes, logger=logger, m=attributes_amount)
            forest.append(tree)

            if img_folder:
                dt.export_dot(tree).to_picture(
                    '{}/forest-{}-tree-{}.png'.format(img_folder.rstrip('/'), f_index + 1, t_index + 1))
            if json_folder:
                with open('{}/forest-{}-tree-{}.json'.format(json_folder.rstrip('/'), f_index + 1, t_index + 1), 'w') as tree_file:
                    tree_file.write(json_exporter.export(tree))

        logger.info('Testing forest with {} instances'.format(len(test)))
        metrics = test_forest(test, forest)
        logger.info('Testing finished')
        logger.info('')
        add_metrics_to_dict(result_dict, metrics,
                            tree_amount, attributes_amount)

        if logger.isEnabledFor(logging.INFO):
            describe_metrics(metrics)

    return result_dict


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
@click.option('--last-layer', '-ll', default=3)
@click.option('--last-neuron', '-ln', default=3)
def execute(filename, separator, k_fold, initial_layer,
            initial_neuron, last_layer, last_neuron):
    """Execute a neural network cross validation"""

    dataset = pd.read_csv(filename, sep=separator)
    y_field = dataset.columns[-1]
    dataset[y_field] = dataset[y_field].astype(str)

    print(dataset)
    print(y_field)

    start = time.time()
    for layer_amount in range(initial_layer, last_layer + 1):
        for neuron_amount in range(initial_neuron, last_neuron + 1):

            net_arch = layer_amount * [neuron_amount]

            print(layer_amount, neuron_amount, net_arch)
            # r = cross_validation(
            #     dataset, k_fold, tree_amount, attributes_amount, img_folder, json_folder)

            # for k in r.keys():
            #     test_results[k].extend(r[k])

            # if test_result_output:
            #     result_frame = pd.DataFrame(test_results)
            #     result_frame.to_csv(test_result_output, index=False)

    end = time.time()

    logger.info('Multiple cross validation finished')
    print()
    print('Time elapsed {} seconds'.format(end - start))


if __name__ == "__main__":
    main()
