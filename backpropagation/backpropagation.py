import os
import logging
import math
import random
import time
import click
import click_log
import pandas as pd
import network as nt

logger = logging.getLogger(__name__)
click_log.basic_config(logger)


           
@click.group()
def main():
    pass


@main.command(name='backpropagation')
@click_log.simple_verbosity_option(logger)
@click.argument('network_filename')
@click.argument('initial_weights_filename')
@click.argument('data_set_filename')

def backpropagation(network_filename, initial_weights_filename, data_set_filename):
    network = nt.Network(network_filename, initial_weights_filename)
    network.print()
    
    data_set = nt.DataSet(data_set_filename, network.num_entries)

    cost = network.cost_regularizaded(data_set)
    print('cost regularizaded = {}'.format(cost))
    


if __name__ == "__main__":
    main()
