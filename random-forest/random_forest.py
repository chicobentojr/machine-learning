import random
import pandas as pd
import click
import decision_tree as dt
from anytree import RenderTree


def get_dataset_fold(d, fold_length):
    return d.sample(fold_length)


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
    accuracy = 0.0
    correct_amount = 0
    for index, predictions in enumerate(predicted_labels):
        if forest_decision(predictions) == real_labels[index]:
            correct_amount += 1
        print(forest_decision(predictions), '\t\t', real_labels[index])

    print()
    print('Accuracy {:0.2f}%'.format(
        100 * (correct_amount / len(real_labels))))


@click.group()
def main():
    pass


@main.command(name='create')
@click.argument('filename')
@click.option('--separator', '-s', default=';', help='your custom csv separator (e.g.: , or ;)')
@click.option('--tree-amount', '-t', default=1, help='...')
@click.option('--attributes-amount', '-a', default=-1, help='...')
def create(filename, separator, tree_amount, attributes_amount):
    """Create a random forest based on a train dataset"""
    dataset = pd.read_csv(filename, sep=separator)

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
