import logging
import random
import pandas as pd
import click
import click_log
import decision_tree as dt
from anytree import RenderTree

logger = logging.getLogger(__name__)
click_log.basic_config(logger)


def get_dataset_fold(d, fold_length):
    return d.sample(fold_length)


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

    print('indexes')
    print(indexes)
    print()
    print('training')
    print(training)
    print()
    print('test')
    print(test)
    print()

    return (training, test)


def forest_decision(trees_predictions):
    most_commom = max(set(trees_predictions), key=trees_predictions.count)
    return most_commom


def test_trees(test_data, trees):
    print(test_data)
    print()
    label_field = test_data.columns[-1]
    real_labels = test_data[label_field].values
    # print(real_labels)
    predicted_labels = []
    for (index, (_, row)) in enumerate(test_data.iterrows()):
        predicted_labels.append([])
        for tree in trees:
            predicted_label = dt.classify_instance(tree, row)
            predicted_labels[index].append(predicted_label)

    # print(predicted_labels)
    print('PREDICTIONS')
    for index in range(len(trees)):
        print('Tree', index + 1, end='\t')
    print()
    for p in predicted_labels:
        print('\t'.join(p))
    print('-'*50)
    print()
    print('Predicted', '\t', 'Real')
    correct_amount = 0
    for index, predictions in enumerate(predicted_labels):
        if forest_decision(predictions) == real_labels[index]:
            correct_amount += 1
        print(forest_decision(predictions), '\t\t', real_labels[index])

    accuracy = correct_amount / len(real_labels)
    print()
    print('Accuracy {:0.2f}%'.format(100 * accuracy))


@click.group()
def main():
    pass


@main.command(name='create')
@click_log.simple_verbosity_option(logger)
@click.argument('filename')
@click.option('--separator', '-s', default=';', help='your custom csv separator (e.g.: , or ;)')
@click.option('--tree-amount', '-t', default=1, help='...')
@click.option('--attributes-amount', '-a', default=-1, help='...')
@click.option('--k-fold', '-k', default=5, help='...')
def create(filename, separator, tree_amount, attributes_amount, k_fold):
    """Create a random forest based on a train dataset"""

    dataset = pd.read_csv(filename, sep=separator)

    dt.generate_tree(dataset, None, logger=logger)

    folds = []
    folds_indexes = []
    fold_test_size = int(len(dataset) / k_fold)
    indexes = dataset.index.tolist()

    random.shuffle(indexes)
    print(indexes)

    for x in range(k_fold - 1):
        test_indexes = indexes[x*fold_test_size:(x+1)*fold_test_size]
        print(test_indexes)
        folds_indexes.append(test_indexes)
        pass
    folds_indexes.append(indexes[(k_fold - 1) * fold_test_size:])

    for test_fold in folds_indexes:
        selected_indexes = dataset.index.isin(test_fold)
        test_set = dataset.iloc[selected_indexes]
        training_set = dataset[~selected_indexes]
        folds.append((training_set, test_set))
    # for x in range(2):
    #     get_dataset_bootstrap(dataset)
    # return

    for train, test in folds:
        print(train)
        print('FOLD...')
        attributes = dataset.columns[:-1].tolist()

        trees = []
        trees_attributes = []
        for i in range(tree_amount):
            print('BOOTSTRAP TREE', i + 1)
            bootstrap_train, _ = get_dataset_bootstrap(train)
            print('BOOTSTRAP TRAIN', i + 1)
            print(bootstrap_train)

            if attributes_amount != -1:
                random.shuffle(attributes)
                attrs = attributes[:attributes_amount]
            else:
                attrs = attributes

            tree = dt.generate_tree(bootstrap_train, attrs)
            trees.append(tree)
            trees_attributes.append(attrs)

        for index, tree in enumerate(trees):
            print('Tree', index + 1)
            print('Attributes')
            print(trees_attributes[index])
            print('-'*50)
            print(RenderTree(tree))
            print('-'*50)
        print()

        print()
        # print(test)
        print()
        print()

    return
    attributes = dataset.columns[:-1].tolist()

    trees = []
    trees_attributes = []

    for index in range(tree_amount):
        if attributes_amount != -1:
            random.shuffle(attributes)
            attrs = attributes[:attributes_amount]
        else:
            attrs = attributes

        tree = dt.generate_tree(dataset, attrs)
        trees.append(tree)
        trees_attributes.append(attrs)

    for index, tree in enumerate(trees):
        print('Tree', index + 1)
        print('Attributes')
        print(trees_attributes[index])
        print('-'*50)
        print(RenderTree(tree))
        print('-'*50)
        print()

    # testing
    test_trees(get_dataset_fold(dataset, 3), trees)


if __name__ == "__main__":
    main()
