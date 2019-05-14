import os
import logging
import math
import random
import pandas as pd
import click
import click_log
import json
import decision_tree as dt
from functools import reduce
from anytree import RenderTree
from anytree.exporter import JsonExporter, DotExporter

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


def get_labels_filtered(labels):
    if len(labels) == 2:
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


def get_pretty_confusion_matrix(matrix_dict, sep='-'):
    labels = get_labels_from_confusion_matrix(matrix_dict, sep)
    m = []
    r = '\t{}\n'.format('\t'.join(labels))
    for index, x_label in enumerate(labels):
        line = '{}\t'.format(x_label)
        m.append([])
        for y_label in labels:
            line += '{}\t'.format(
                matrix_dict['{}{}{}'.format(x_label, sep, y_label)])
            m[index].append(
                matrix_dict['{}{}{}'.format(x_label, sep, y_label)])
        r += '{}\n'.format(line)
    print(r)
    for n in m:
        print(n)
    return r


def get_evaluation_metrics(matrix, sep='-'):
    labels = get_labels_from_confusion_matrix(matrix, sep)
    n = reduce((lambda x, y: x + y), matrix.values())
    logger.debug('Total predicted: {}'.format(n))
    metrics = {
        ACCURACY: 0,
        PRECISION_MACRO: 0,
        PRECISION_MICRO: 0,
        RECALL_MACRO: 0,
        RECALL_MICRO: 0,
        LABELS: {}
    }
    total_vp = 0
    total_fp = 0
    total_fn = 0

    for x in labels:
        metrics[ACCURACY] += matrix['{}{}{}'.format(x, sep, x)]
        metrics[LABELS][x] = {'vp': 0, 'fp': 0, 'fn': 0}

        for y in labels:
            if x == y:
                metrics[LABELS][x]['vp'] = matrix['{}{}{}'.format(x, sep, y)]
            else:
                metrics[LABELS][x]['fp'] = matrix['{}{}{}'.format(y, sep, x)]
                metrics[LABELS][x]['fn'] = matrix['{}{}{}'.format(x, sep, y)]
        vp = metrics[LABELS][x]['vp']
        fn = metrics[LABELS][x]['fn']
        fp = metrics[LABELS][x]['fp']

        total_vp += vp
        total_fp += fp
        total_fn += fn

        metrics[LABELS][x][PRECISION] = vp / (vp + fp) if vp > 0 else 0
        metrics[LABELS][x][RECALL] = vp / (vp + fn) if vp > 0 else 0

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


def forest_decision(trees_predictions):
    most_commom = max(set(trees_predictions), key=trees_predictions.count)
    return most_commom


def get_f_measure(precision, recall, beta=1.0):
    return (1 + math.pow(beta, 2)) * ((precision * recall)/((math.pow(beta, 2) * precision) + recall))


def processing_result_metrics(metrics):
    accuracy = metrics[ACCURACY]
    print()
    print('Accuracy {}'.format(accuracy))
    print('Error rate {}'.format(1 - accuracy))

    for label in metrics[LABELS].keys():
        print('Processing {}'.format(label))

        vp = metrics[LABELS][label]['vp']
        fn = metrics[LABELS][label]['fn']
        fp = metrics[LABELS][label]['fp']

        precision = vp / (vp + fp) if vp > 0 else 0
        recall = vp / (vp + fn) if vp > 0 else 0

        print('\tPrecision: {}'.format(precision))
        print('\tRecall: {}'.format(recall))
        print('\tF-Measure: {}'.format(get_f_measure(precision, recall)))
        print()


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

    confusion_matrix = {}
    possible_labels = get_labels_filtered(unique_labels)
    for x_label in possible_labels:
        for y_label in possible_labels:
            confusion_matrix['{}-{}'.format(x_label, y_label)] = 0

    for index, predictions in enumerate(predicted_labels):
        prediction = forest_decision(predictions)
        real_label = real_labels.values[index]
        if '{}-{}'.format(real_label, predicted_label) in confusion_matrix:
            confusion_matrix['{}-{}'.format(real_label, prediction)] += 1

        logger.debug('{}\t\t\t{}'.format(prediction, real_label))

    metrics = get_evaluation_metrics(confusion_matrix)
    logger.debug('Confusion Matrix')
    logger.debug(get_pretty_confusion_matrix(confusion_matrix))
    logger.debug('')
    logger.debug('Metrics')
    logger.debug(json.dumps(metrics, indent=2))
    return metrics


@click.group()
def main():
    pass


@main.command(name='create')
@click_log.simple_verbosity_option(logger)
@click.argument('filename')
@click.option('--separator', '-s', default=',', help='your custom csv separator (e.g.: ; or :)')
@click.option('--tree-amount', '-t', default=1, help='your tree amount per forest')
@click.option('--attributes-amount', '-a', default=0, help='your attributes number to generate each tree')
@click.option('--k-fold', '-k', default=5, help='your number of folds to cross validation')
@click.option('--json-folder', '-json', default='', help='your folder to save result trees in JSON format')
@click.option('--img-folder', '-img', default='', help='your folder to save result trees in PNG format')
@click.option('--bootstrap', is_flag=True)
def create(filename, separator, tree_amount, attributes_amount, k_fold, json_folder, img_folder, bootstrap):
    """Create a random forest based on a CSV dataset"""

    dataset = pd.read_csv(filename, sep=separator)
    attributes = dataset.columns[:-1].tolist()

    if json_folder:
        json_exporter = JsonExporter(indent=2)
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)

    if img_folder:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

    if attributes_amount == 0:
        attributes_amount = int(math.sqrt(len(attributes)))

    if bootstrap:
        logger.debug('BOOTSTRAP')
        logger.debug('Dataset')
        logger.debug(dataset)
        logger.debug('')
        train, test = get_dataset_bootstrap(dataset)
        logger.debug('train')
        logger.debug(train)
        logger.debug('')
        logger.debug('test')
        logger.debug(test)
        logger.debug('')

        forest = []
        trees_attributes = []
        for i in range(tree_amount):
            if attributes_amount != -1:
                random.shuffle(attributes)
                attrs = attributes[:attributes_amount]
            else:
                attrs = attributes

            tree = dt.generate_tree(train, attrs, logger=logger)
            forest.append(tree)
            trees_attributes.append(attrs)

            if img_folder:
                DotExporter(tree).to_picture(
                    '{}/bootstrap-tree-{}.png'.format(img_folder.rstrip('/'), i + 1))

        metrics = test_forest(test, forest)
        logger.info('Metrics')
        logger.info(json.dumps(metrics, indent=2))
        processing_result_metrics(metrics)

    else:
        folds = generate_folds(dataset, k_fold)

        for fold_index, (train, test) in enumerate(folds):
            forest = []
            trees_attributes = []
            logger.info('Processing fold {}'.format(fold_index + 1))
            logger.info('Training...')
            for i in range(tree_amount):
                bootstrap_train, _ = get_dataset_bootstrap(train)

                if attributes_amount != -1:
                    random.shuffle(attributes)
                    attrs = attributes[:attributes_amount]
                else:
                    attrs = attributes

                tree = dt.generate_tree(bootstrap_train, attrs, logger=logger)
                forest.append(tree)
                trees_attributes.append(attrs)

                if img_folder:
                    DotExporter(tree).to_picture(
                        '{}/forest-{}-tree-{}.png'.format(img_folder.rstrip('/'), fold_index + 1, i + 1))

            for index, tree in enumerate(forest):
                logger.debug(f'Tree {index+1}')
                logger.debug('Attributes')
                logger.debug(trees_attributes[index])
                logger.debug('-'*50)
                logger.debug(RenderTree(tree))
                logger.debug('-'*50)
                if json_folder:
                    with open('{}/forest-{}-tree-{}.json'.format(json_folder.rstrip('/'), fold_index + 1, index + 1), 'w') as tree_file:
                        tree_file.write(json_exporter.export(tree))

            logger.info('Testing ...')
            metrics = test_forest(test, forest)
            logger.info('Test result')
            processing_result_metrics(metrics)


if __name__ == "__main__":
    main()
