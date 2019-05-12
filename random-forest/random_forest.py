import logging
import math
import random
import pandas as pd
import click
import click_log
import decision_tree as dt
from anytree import RenderTree

logger = logging.getLogger(__name__)
click_log.basic_config(logger)


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


def test_trees(test_data, trees):
    label_field = test_data.columns[-1]
    real_labels = test_data[label_field].values

    predicted_labels = []
    for (index, (_, row)) in enumerate(test_data.iterrows()):
        predicted_labels.append([])
        for tree in trees:
            predicted_label = dt.classify_instance(tree, row)
            predicted_labels[index].append(predicted_label)

    logger.info('PREDICTIONS')
    logger.info('')
    logger.info('\t'.join(['Tree {}'.format(i+1) for i in range(len(trees))]))
    for p in predicted_labels:
        logger.info('\t'.join(p))
    logger.info('-'*50)
    logger.info('')
    logger.info('Forest Predicted\tReal')

    correct_amount = 0

    for index, predictions in enumerate(predicted_labels):
        if forest_decision(predictions) == real_labels[index]:
            correct_amount += 1
        logger.info('{}\t\t\t{}'.format(
            forest_decision(predictions), real_labels[index]))

    accuracy = correct_amount / len(real_labels)
    logger.info('')
    logger.info('Accuracy {:0.2f}%'.format(100 * accuracy))
    logger.info('')


@click.group()
def main():
    pass


@main.command(name='create')
@click_log.simple_verbosity_option(logger)
@click.argument('filename')
@click.option('--separator', '-s', default=';', help='your custom csv separator (e.g.: , or ;)')
@click.option('--tree-amount', '-t', default=1, help='your tree amount per forest')
@click.option('--attributes-amount', '-a', default=0, help='your attributes number to generate each tree')
@click.option('--k-fold', '-k', default=5, help='your number of folds to cross validation')
def create(filename, separator, tree_amount, attributes_amount, k_fold):
    """Create a random forest based on a CSV dataset"""

    dataset = pd.read_csv(filename, sep=separator)

    if attributes_amount == 0:
        attributes_amount = int(math.sqrt(attributes_amount))

    folds = generate_folds(dataset, k_fold)

    for train, test in folds:
        attributes = dataset.columns[:-1].tolist()

        trees = []
        trees_attributes = []
        for i in range(tree_amount):
            bootstrap_train, _ = get_dataset_bootstrap(train)

            if attributes_amount != -1:
                random.shuffle(attributes)
                attrs = attributes[:attributes_amount]
            else:
                attrs = attributes

            tree = dt.generate_tree(bootstrap_train, attrs, logger=logger)
            trees.append(tree)
            trees_attributes.append(attrs)

        for index, tree in enumerate(trees):
            logger.debug(f'Tree {index+1}')
            logger.debug('Attributes')
            logger.debug(trees_attributes[index])
            logger.debug('-'*50)
            logger.debug(RenderTree(tree))
            logger.debug('-'*50)

        test_trees(test, trees)


if __name__ == "__main__":
    main()
