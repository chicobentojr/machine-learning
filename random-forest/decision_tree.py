import pandas as pd
from anytree import Node, RenderTree, AnyNode

NODE_TYPE_DECISION = 'D'
NODE_TYPE_LABEL = 'L'


def create_decision_tree(filename, separator=';'):
    dataset = pd.read_csv(filename, sep=separator)

    ys = dataset.iloc[:, len(dataset.columns) - 1]
    d = dataset
    attrs = dataset.columns[:-1]

    for attr in attrs:
        print(attr)

        for value in d[attr].unique():
            nd = []
            print(attr, value)
            di = d[d.apply(lambda x: x[attr] == value, axis=1)]
            print(di)
            print(di.index.tolist())
            print(ys[di.index.tolist()].tolist())

        print()
        # print(dataset[attribute])


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
