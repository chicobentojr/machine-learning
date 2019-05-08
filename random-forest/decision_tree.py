import pandas as pd
import math
from anytree import Node, RenderTree, AnyNode

NODE_TYPE_DECISION = 'D'
NODE_TYPE_LABEL = 'L'


def info_dataset(total, class_amount):
    return - (class_amount/total) * math.log2(class_amount/total)


def generate_tree(d, parent=None, parent_value=None):
    print('DATASET')
    print(d)
    print('-'*50)
    print()
    original_entropy = 0.0
    y_field = d.columns[len(d.columns) - 1]
    original_classes = d[y_field].unique()
    attrs = d.columns[:-1]

    if len(original_classes) == 1:
        return AnyNode(parent, type=NODE_TYPE_LABEL, label=original_classes[0], value=parent_value)

    for c in original_classes:
        d_c = d[d.apply(lambda x: x[y_field] == c, axis=1)]
        original_entropy += info_dataset(len(d), len(d_c))

    print('Original Entropy', original_entropy)
    print()

    choosed_field = ''
    choosed_gain = 0.0

    for attr in attrs:
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

        if attr_gain > choosed_gain:
            choosed_gain, choosed_field = attr_gain, attr
        print()

    print()
    print('Choosed field', choosed_field)
    print('Choosed Gain', choosed_gain)
    print()

    decision = AnyNode(parent,
                       type=NODE_TYPE_DECISION,
                       field=choosed_field, value=parent_value)

    for v in d[choosed_field].unique():
        print(choosed_field, v)
        new_d = d[d.apply(lambda x: x[choosed_field] == v, axis=1)]
        new_d = new_d.drop(choosed_field, 1)
        generate_tree(new_d, decision, v)

    return decision


def create_decision_tree(filename, separator=';'):
    dataset = pd.read_csv(filename, sep=separator)

    d = dataset

    root = generate_tree(d)

    print(RenderTree(root))

#     print(dataset)
#     print('-'*20)
#     print('Summary')
#     print(dataset.describe())
#     print('-'*20)

#     root = Node('root')
#     tempo = Node('tempo', parent=root)
#     temp = Node('temperatura', parent=root)
#     jane = AnyNode(parent=root, field='Tempo')
#     leaf = AnyNode(parent=jane, label='Não')

#     print(jane)
#     print(jane.field)

#     print()
#     print(RenderTree(root))
#     print()
#     print(dataset.columns)

#     attriutes = dataset.columns

#     Y = dataset[attriutes[len(attriutes) - 1]]
#     print(Y)
#     print()
#     print(dataset.iloc[0, :])
#     print()


if __name__ == "__main__":
    create_decision_tree('datasets/benchmark.csv')
