import pandas as pd
import math
import logging
import click
import click_log
from pandas.api.types import is_string_dtype, is_numeric_dtype
from anytree import Node, RenderTree, AnyNode
from anytree.dotexport import DotExporter

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

NODE_TYPE_DECISION = 'D'
NODE_TYPE_LABEL = 'L'
NODE_NAME_SEPARATOR = '\n'

def node_attr_func(node):
    parts = node.name.split(NODE_NAME_SEPARATOR)
    return 'label="%s"' % (parts[-1])
    
def edge_attr_func(node, child):
    return 'label="%s"' % (str(child.value))

def edge_type_func(node, child):
    return '--'

def export_dot(tree):
    return DotExporter(tree, graph="graph",
                       nodeattrfunc=node_attr_func,
                       edgeattrfunc=edge_attr_func,
                       edgetypefunc=edge_type_func)


def info_dataset(total, class_amount):
    return - (class_amount/total) * math.log2(class_amount/total)


def get_attribute_split_node():
    return 0


def get_numeric_attribute_split(d, attr):
    return d[attr].mean()


def get_attribute_entropy(d, attr):
    attr_entropy = 0.0
    y_field = d.columns[-1]

    # attribute is string
    if pd.api.types.is_string_dtype(d[attr]):
        for value in d[attr].unique():
            di = d[d.apply(lambda x: x[attr] == value, axis=1)]
            possible_classes = di[y_field].unique()

            di_total = len(di)
            di_info = 0.0

            for c in possible_classes:
                di_c = di[di.apply(lambda x: x[y_field] == c, axis=1)]
                di_c_amount = len(di_c)

                di_info += info_dataset(len(di), di_c_amount)

            attr_entropy += (di_total / len(d)) * di_info

    # attribute is numeric
    if pd.api.types.is_numeric_dtype(d[attr]):
        min = d[attr].min()
        middle = get_numeric_attribute_split(d, attr)
        max = d[attr].max()

        for minimum, maximum in [(min - 1, middle), (middle, max)]:
            di = d[d.apply(lambda x: minimum < x[attr] <= maximum, axis=1)]
            possible_classes = di[y_field].unique()

            di_total = len(di)
            di_info = 0.0

            for c in possible_classes:
                di_c = di[di.apply(lambda x: x[y_field] == c, axis=1)]
                di_c_amount = len(di_c)

                di_info += info_dataset(len(di), di_c_amount)

            attr_entropy += (di_total / len(d)) * di_info

    return attr_entropy


def generate_tree(d, attributes, parent=None, parent_value=None, verbose=False, logger=logger):
    logger.debug('DATASET')
    logger.debug(d)
    logger.debug('-'*50)
    logger.debug('SUMMARY')
    logger.debug(d.describe())
    logger.debug('-'*50)

    original_entropy = 0.0
    y_field = d.columns[len(d.columns) - 1]
    original_classes = d[y_field].unique()
    # attributes = d.columns[:-1].tolist()
    most_frequent_label = d[y_field].max()

    logger.debug('COLUMNS')
    logger.debug(attributes)
    logger.debug('-'*50)
    logger.debug('')

    if len(original_classes) == 1:
        return AnyNode(parent, type=NODE_TYPE_LABEL, label=original_classes[0], value=parent_value,
                       name='{}= {}'.format(parent_value + NODE_NAME_SEPARATOR if isinstance(parent_value, str) else '', original_classes[0]))
    elif not attributes:
        return AnyNode(parent, type=NODE_TYPE_LABEL, label=most_frequent_label, value=parent_value,
                       name='{}= {}'.format(parent_value + NODE_NAME_SEPARATOR if isinstance(parent_value, str) else '', most_frequent_label))

    for c in original_classes:
        d_c = d[d.apply(lambda x: x[y_field] == c, axis=1)]
        original_entropy += info_dataset(len(d), len(d_c))

    logger.debug('Original Entropy {}'.format(original_entropy))
    logger.debug('')

    chosen_field = attributes[0]
    higher_gain = 0.0

    for attr in attributes:
        logger.debug(attr)
        attr_entropy = get_attribute_entropy(d, attr)

        attr_gain = original_entropy - attr_entropy
        logger.debug('Entropy {} = {}'.format(attr, attr_entropy))
        logger.debug('Gain {} = {}'.format(attr, attr_gain))

        if attr_gain > higher_gain:
            higher_gain, chosen_field = attr_gain, attr
        logger.debug('')

    logger.debug('')
    logger.debug('Higher Gain = {}'.format(higher_gain))
    logger.debug('The chosen field is "{}"'.format(chosen_field))
    logger.debug('')

    # attributes.remove(chosen_field)
    attributes = [x for x in attributes if x != chosen_field]

    if pd.api.types.is_string_dtype(d[chosen_field]):
        decision = AnyNode(parent,
                           type=NODE_TYPE_DECISION,
                           name='{}{}{}?'.format(
                               parent_value + NODE_NAME_SEPARATOR if parent_value else '',
                               NODE_NAME_SEPARATOR,
                               chosen_field),
                           field=chosen_field, value=parent_value,
                           label=most_frequent_label)

        for v in d[chosen_field].unique():
            logger.debug('{} {}'.format(chosen_field, v))
            new_d = d[d.apply(lambda x: x[chosen_field] == v, axis=1)]
            new_d = new_d.drop(chosen_field, 1)
            generate_tree(new_d, attributes, decision, v)

    elif pd.api.types.is_numeric_dtype(d[chosen_field]):
        number = get_numeric_attribute_split(d, chosen_field)
        decision = AnyNode(parent,
                           type=NODE_TYPE_DECISION,
                           name='{}{}{} <= {}?'.format(
                               parent_value + NODE_NAME_SEPARATOR if parent_value else '',
                               NODE_NAME_SEPARATOR,
                               chosen_field, number),
                           field=chosen_field, value=parent_value,
                           label=most_frequent_label)

        min = d[chosen_field].min()
        middle = get_numeric_attribute_split(d, chosen_field)
        max = d[chosen_field].max()

        for minimum, maximum in [(min - 1, middle), (middle, max)]:
            logger.debug('{} {} {}'.format(chosen_field, minimum, maximum))
            new_d = d[d.apply(lambda x: minimum <
                              x[chosen_field] <= maximum, axis=1)]
            new_d = new_d.drop(chosen_field, 1)
            generate_tree(new_d, attributes, decision, middle)

    return decision


def create_decision_tree(filename, separator=';'):
    dataset = pd.read_csv(filename, sep=separator)
    root = generate_tree(dataset, dataset.columns[:-1].tolist())
    return root


def classify_instance(decision_tree, instance):
    node = decision_tree
    while node.type != NODE_TYPE_LABEL:
        v = instance[node.field]

        if isinstance(v, float) or isinstance(v, int):
            if v <= node.children[0].value:
                node = node.children[0]
            else:
                node = node.children[1]
            break
        else:
            for child in node.children:
                if v == child.value:
                    node = child
                    break
            else:
                return node.label

    return node.label


@click.group()
def main():
    pass


@main.command(name='create')
@click_log.simple_verbosity_option(logger)
@click.argument('filename')
@click.option('--separator', '-s', default=';', help='your custom csv separator (e.g.: , or ;)')
@click.option('--image-output', '-img', help='a filename to storage the result decision tree.\nNeeds graphviz installed')

def create(filename, separator, image_output):
    """Create a decision tree based on a train dataset"""
    decision_tree = create_decision_tree(filename, separator=separator)

    logger.debug(RenderTree(decision_tree))
    logger.debug('')

    if image_output:
        if not image_output.endswith('.png'):
            image_output += '.png'
        export_dot(decision_tree).to_picture(image_output)
    
    test_instances = pd.DataFrame({
        'Tempo': ['Ensolarado', 'Chuvoso'],
        'Temperatura': ['Quente', 'Fria'],
        'Umidade': ['Normal', 'Alta'],
        'Ventoso': ['Verdadeiro', 'Verdadeiro'],
        'Jogar': ['?', '?']
    })

    for idx, instance in test_instances.iterrows():
        logger.debug('Instance {}\n{}'.format(instance.index, instance.values))
        logger.debug('Result {}'.format(
            classify_instance(decision_tree, instance)))


if __name__ == "__main__":
    main()