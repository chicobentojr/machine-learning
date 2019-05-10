import pandas as pd
import math
import click
from anytree import Node, RenderTree, AnyNode
from anytree.dotexport import DotExporter

NODE_TYPE_DECISION = 'D'
NODE_TYPE_LABEL = 'L'


def info_dataset(total, class_amount):
    return - (class_amount/total) * math.log2(class_amount/total)


def generate_tree(d, attributes, parent=None, parent_value=None):
    print('DATASET')
    print(d)
    print('-'*50)

    original_entropy = 0.0
    y_field = d.columns[len(d.columns) - 1]
    original_classes = d[y_field].unique()
    attributes = d.columns[:-1].tolist()

    print('COLUMNS')
    print(attributes)
    print('-'*50)
    print()

    if len(original_classes) == 1:
        return AnyNode(parent, type=NODE_TYPE_LABEL, label=original_classes[0], value=parent_value,
                       name=parent_value + '\n= ' + original_classes[0])
    elif not attributes:
        most_frequent_label = d[y_field].max()
        return AnyNode(parent, type=NODE_TYPE_LABEL, label=most_frequent_label, value=parent_value,
                       name=parent_value + '\n= ' + most_frequent_label)

    for c in original_classes:
        d_c = d[d.apply(lambda x: x[y_field] == c, axis=1)]
        original_entropy += info_dataset(len(d), len(d_c))

    print('Original Entropy', original_entropy)
    print()

    chosen_field = ''
    higher_gain = 0.0

    for attr in attributes:
        print(attr)
        attr_entropy = 0.0

        for value in d[attr].unique():
            di = d[d.apply(lambda x: x[attr] == value, axis=1)]

            possible_classes = di.iloc[:, len(d.columns) - 1].unique()
            di_total = len(di)
            di_info = 0.0

            for c in possible_classes:
                di_c = di[di.apply(lambda x: x[y_field] == c, axis=1)]
                di_c_amount = len(di_c)

                di_info += info_dataset(len(di), di_c_amount)

            attr_entropy += (di_total / len(d)) * di_info

        attr_gain = original_entropy - attr_entropy
        print('Entropy', attr, '=', attr_entropy)
        print('Gain', attr, '=', attr_gain)

        if attr_gain > higher_gain:
            higher_gain, chosen_field = attr_gain, attr
        print()

    print()
    print('Higher Gain', '=', higher_gain)
    print('The chosen field is "{}"'.format(chosen_field))
    print()

    decision = AnyNode(parent,
                       type=NODE_TYPE_DECISION,
                       name=str(parent_value) + '\n' + chosen_field + '?',
                       field=chosen_field, value=parent_value)

    # attributes.remove(chosen_field)

    for v in d[chosen_field].unique():
        print(chosen_field, v)
        new_d = d[d.apply(lambda x: x[chosen_field] == v, axis=1)]
        new_d = new_d.drop(chosen_field, 1)
        generate_tree(new_d, attributes, decision, v)

    return decision


@click.group()
def main():
    pass


@main.command(name='create')
@click.argument('filename')
@click.option('--separator', '-s', default=';', help='your custom csv separator (e.g.: , or ;)')
@click.option('--image-output', '-img', help='a filename to storage the result decision tree.\nNeeds graphviz installed')
def create(filename, separator, image_output):
    """Create a decision tree based on a train dataset"""
    decision_tree = create_decision_tree(filename, separator=separator)

    print(RenderTree(decision_tree))
    print()

    if image_output:
        if not image_output.endswith('.png'):
            image_output += '.png'
        DotExporter(decision_tree).to_picture(image_output)

    test_instances = pd.DataFrame({
        'Tempo': ['Ensolarado', 'Chuvoso'],
        'Temperatura': ['Quente', 'Fria'],
        'Umidade': ['Normal', 'Alta'],
        'Ventoso': ['Verdadeiro', 'Verdadeiro'],
        'Jogar': ['?', '?']
    })

    for idx, instance in test_instances.iterrows():
        print('Instance', instance.index, '\n', instance.values)
        print('Result', classify_instance(decision_tree, instance))


def create_decision_tree(filename, separator=';'):
    dataset = pd.read_csv(filename, sep=separator)
    root = generate_tree(dataset, dataset.columns[:-1].tolist())
    return root


def classify_instance(decision_tree, instance):
    node = decision_tree
    while node.type != NODE_TYPE_LABEL:
        v = instance[node.field]
        for child in node.children:
            if v == child.value:
                node = child
                break

    return node.label


if __name__ == "__main__":
    main()
