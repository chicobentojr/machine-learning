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
NODE_TYPE_LEAF = 'L'
NODE_NAME_SEPARATOR = '\n'


def node_attr_func(node):
    parts = node.name.split(NODE_NAME_SEPARATOR)
    label = parts[-1].replace("= ", "")
    return 'label="%s"' % (label)


def edge_attr_func(node, child):
    return 'label="%s"' % (str(child.value))


def edge_type_func(node, child):
    return '--'


def export_dot(tree):
    return DotExporter(tree, graph="graph",
                       nodeattrfunc=node_attr_func,
                       edgeattrfunc=edge_attr_func,
                       edgetypefunc=edge_type_func)


def get_dataset_entropy(total, class_amount):
    return - (class_amount/total) * math.log2(class_amount/total)


def get_attribute_split_node():
    return 0


def get_numeric_attribute_split(d, attr):
    return round(d[attr].mean(), 3)


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

                di_info += get_dataset_entropy(len(di), di_c_amount)

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

                di_info += get_dataset_entropy(len(di), di_c_amount)

            attr_entropy += (di_total / len(d)) * di_info

    return attr_entropy


def generate_tree(d, attributes, parent=None, parent_value=None, verbose=False, logger=logger, i=0):
    logger.debug('DATASET')
    logger.debug(d)
    logger.debug('-'*50)
    logger.debug('')
    logger.debug('SUMMARY')
    logger.debug(d.describe())
    logger.debug('-'*50)

    original_entropy = 0.0
    y_field = d.columns[len(d.columns) - 1]
    original_classes = d[y_field].unique()
    most_frequent_label = d[y_field].mode()[0] if not d.empty else parent.label

    logger.debug('MOST FREQUENT LABEL {}'.format(most_frequent_label))
    logger.debug('')
    logger.debug('COLUMNS')
    logger.debug(attributes)
    logger.debug('-'*50)
    logger.debug('')

    if len(original_classes) == 1:
        return AnyNode(parent,
                       type=NODE_TYPE_LEAF,
                       label=original_classes[0],
                       value=parent_value,
                       name='{}{}= {}'.format(i, parent_value + NODE_NAME_SEPARATOR if isinstance(parent_value, str) else NODE_NAME_SEPARATOR,
                                              original_classes[0]))
    elif not attributes or d.empty:
        return AnyNode(parent,
                       type=NODE_TYPE_LEAF,
                       label=most_frequent_label,
                       value=parent_value,
                       name='{}{}= {}'.format(i, parent_value + NODE_NAME_SEPARATOR if isinstance(parent_value, str) else NODE_NAME_SEPARATOR,
                                              most_frequent_label))

    for c in original_classes:
        d_c = d[d.apply(lambda x: x[y_field] == c, axis=1)]
        original_entropy += get_dataset_entropy(len(d), len(d_c))

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

    # attributes = [x for x in attributes if x != chosen_field]

    if pd.api.types.is_string_dtype(d[chosen_field]):
        decision = AnyNode(parent,
                           type=NODE_TYPE_DECISION,
                           label=most_frequent_label,
                           value=parent_value,
                           name='{}{}{}{}?'.format(i,
                                                   str(parent_value) +
                                                   NODE_NAME_SEPARATOR if parent_value else NODE_NAME_SEPARATOR,
                                                   NODE_NAME_SEPARATOR,
                                                   chosen_field),
                           field=chosen_field)

        for v in d[chosen_field].unique():
            logger.debug('Chosen field {} value = {}'.format(chosen_field, v))
            new_d = d[d.apply(lambda x: x[chosen_field] == v, axis=1)]
            # new_d = new_d.drop(chosen_field, 1)
            generate_tree(new_d, attributes, decision, v, i=i+1)
            i = i + 1

    elif pd.api.types.is_numeric_dtype(d[chosen_field]):
        number = get_numeric_attribute_split(d, chosen_field)
        logger.debug('Chosen field {} split value = {}'.format(
            chosen_field, number))

        decision = AnyNode(parent,
                           type=NODE_TYPE_DECISION,
                           name='{}{}{}{} <= {}?'.format(i,
                                                         str(parent_value) +
                                                         NODE_NAME_SEPARATOR if parent_value else NODE_NAME_SEPARATOR,
                                                         NODE_NAME_SEPARATOR,
                                                         chosen_field, number),
                           field=chosen_field, value=parent_value,
                           label=most_frequent_label)

        min = d[chosen_field].min()
        middle = get_numeric_attribute_split(d, chosen_field)
        max = d[chosen_field].max()

        for minimum, maximum in [(min - 1, middle), (middle, max)]:
            logger.debug('Chosen field {} min = {} | max = {}'.format(
                chosen_field, minimum, maximum))
            new_d = d[d.apply(lambda x: minimum <
                              x[chosen_field] <= maximum, axis=1)]
            # new_d = new_d.drop(chosen_field, 1)
            generate_tree(new_d, attributes, decision,
                          middle, logger=logger, i=i+1)
            i = i + 1

    return decision


def create_decision_tree(filename, separator=';'):
    dataset = pd.read_csv(filename, sep=separator)
    root = generate_tree(dataset, dataset.columns[:-1].tolist())
    return root


def classify_instance(decision_tree, instance):
    node = decision_tree  # node = tree root
    while node.type != NODE_TYPE_LEAF:  # NODE_TYPE_LEAF = leaf node
        v = instance[node.field]  # v = instance value for node field

        if isinstance(v, float) or isinstance(v, int):  # if is a numeric attribute
            if v <= node.children[0].value:  # cut value: left value
                node = node.children[0]  # search at left sub-tree
            else:
                node = node.children[1]  # search at right sub-tree
        else:  # is a categoric attribute
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
@click.option('--separator', '-s', default=',', help='your custom csv separator (e.g.: ; or :)')
@click.option('--image-output', '-img', help='a filename to storage the result decision tree.\nNeeds graphviz installed')
def create(filename, separator, image_output):
    """Create a decision tree based on a train dataset"""
    decision_tree = create_decision_tree(filename, separator=separator)

    logger.debug(RenderTree(decision_tree))
    logger.debug('')

    if image_output:
        if not image_output.endswith('.png'):
            image_output += '.png'
        dot = export_dot(decision_tree)
        dot.to_picture(image_output)


if __name__ == "__main__":
    main()
