import pandas as pd
import math
from anytree import Node, RenderTree, AnyNode

NODE_TYPE_DECISION = 'D'
NODE_TYPE_LABEL = 'L'


def info_dataset(total, class_amount):
    return - (class_amount/total) * math.log2(class_amount/total)


def generate_tree(d, attributes, parent=None, parent_value=None):
    print('DATASET')
    print(d)
    print('-'*50)
    print('COLUMNS')
    print(attributes)
    print('-'*50)
    print()
    original_entropy = 0.0
    y_field = d.columns[len(d.columns) - 1]
    original_classes = d[y_field].unique()

    if len(original_classes) == 1:
        return AnyNode(parent, type=NODE_TYPE_LABEL, label=original_classes[0], value=parent_value)

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
                       field=chosen_field, value=parent_value)

    attributes.remove(chosen_field)

    for v in d[chosen_field].unique():
        print(chosen_field, v)
        new_d = d[d.apply(lambda x: x[chosen_field] == v, axis=1)]
        new_d = new_d.drop(chosen_field, 1)
        generate_tree(new_d, attributes, decision, v)

    return decision


def create_decision_tree(filename, separator=';'):
    dataset = pd.read_csv(filename, sep=separator)
    root = generate_tree(dataset, dataset.columns[:-1].tolist())

    print(RenderTree(root))


if __name__ == "__main__":
    create_decision_tree('datasets/benchmark.csv')
