import os
import logging
import math
import random
import time
import click
import click_log
import network_classes as ntc
import numpy
import matrix

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

def cost(xi, yi, f_xi):
    _yi = list(map(lambda y: -y, yi))
    log = list(map(lambda f: math.log10(f), f_xi))
    Ji = numpy.dot(_yi, log)

    n_yi  = list(map(lambda y: 1-y, yi))
    n_log = list(map(lambda f: math.log10(1-f), f_xi))
    Ji -= numpy.dot(n_yi, n_log)
    return Ji

           
@click.group()
def main():
    pass


@main.command(name='backpropagation')
@click_log.simple_verbosity_option(logger)
@click.argument('network_filename')
@click.argument('initial_weights_filename')
@click.argument('data_set_filename')
@click.option('--alpha', '-a', default=0.5, help='Weights Update Rate, is used to smooth the gradient')

def backpropagation(network_filename, initial_weights_filename, data_set_filename,
                    alpha):
    
    network = ntc.Network(network_filename, initial_weights_filename)
    network.print_all()
    
    data_set = ntc.DataSet(data_set_filename, network.num_entries)

    numExamples = len(data_set.examples)
    numLayers = network.total_layers
    delta = [None] * numLayers

    less_acceptable_difference = 0.1 # how much?????
    last_regularized_cost = 1000000

    stop = False
    count = 0

    while not stop and count < 1:
        J = 0 # to define stop
        #logger.info('\nITERATION #{} **********************'.format(count+1))

        logger.info('\n 1. Percorrer exemplos (x,y):')
        
        for i in range(0, numExamples):
            logger.info('\n Exemplo #{}:'.format(i+1))
            logger.info('\n 1.1. Propagacao pela rede:')

            example = data_set.examples[i]
            xi = example.x
            yi = example.y
            f_xi = network.propagate_instance(xi)
            Ji = cost(xi, yi, f_xi)
            Ji_reg = network.regularize_cost(Ji, 1)
            J += numpy.sum(Ji)

            logger.info('\tSaida predita  para o exemplo {}: {}'.format(i+1, ntc.format_list(f_xi)))
            logger.info('\tSaida esperada para o exemplo {}: {}'.format(i+1, ntc.format_list(yi)))
            logger.info('\tJ do exemplo {}: {}'.format(i+1, Ji))
            logger.info('\tJ do exemplo {}: {} (reg)'.format(i+1, Ji_reg))

            delta[-1] = numpy.subtract(f_xi, yi)
            logger.info('\n 1.2. Delta da camada de saida\n\tdelta{} = {}'.format(numLayers, delta[-1]))
            logger.info('\n 1.3. Deltas das camadas ocultas')
            
            for k in range(numLayers-2, 1-1, -1):
                layer_k = network.layers[k]
                theta_k = layer_k.weight_matrix
                a_k = layer_k.neurons
                _1_a_k = list(map(lambda x: 1-x, a_k))
                product = theta_k.transpose_and_multiply_by_vector(delta[k+1])
                delta[k] = numpy.dot(numpy.dot(product, a_k), _1_a_k)
                delta[k] = delta[k][1:len(delta[k])] # remove bias
                logger.info('\tdelta{} = {}'.format(k+1, delta[k]))

            logger.info('\n 1.4. Gradientes dos pesos de cada camada com base nos exemplo atual')

            for k in range(numLayers-2, 0-1, -1):
                layer_k = network.layers[k]
                a_k = layer_k.neurons
                network.layers[k].gradient_matrix.matrix += numpy.matmul(
                    numpy.matrix(delta[k+1]).transpose(),
                    numpy.matrix(a_k))
                
                logger.info('\n\tGradiente de Theta{} = \n{}'.format(k+1, network.layers[k].gradient_matrix.str_tabs(2)))

        # end for

        J /= numExamples
        J = network.regularize_cost(J, numExamples)
        '''
        logger.info('\nCheck peformance to set stop:')
        diff = last_regularized_cost - J
        stop = (diff <= less_acceptable_difference)
        logger.info('diff = last_cost({}) - actual_cost({}) = {}'.format(last_regularized_cost, J, diff))
        last_regularized_cost = J
        '''
        count += 1

        logger.info('\n\n 2. Gradientes finais (regularizados) para os pesos de cada camada')

        for k in range(numLayers-2, 0-1, -1):
            layer_k = network.layers[k]
            theta_k = layer_k.weight_matrix
            P_k = theta_k.matrix * network.regularizationFactor
            D_k = layer_k.gradient_matrix.matrix
            network.layers[k].gradient_matrix.matrix =  (D_k + P_k) / numExamples
            logger.info('\n\tGradiente de Theta{} = \n{}'.format(k+1, network.layers[k].gradient_matrix.str_tabs(2)))

        logger.info('\n 3. Atualizar pesos de cada camada com base nos gradientes')

        for k in range(numLayers-2, 0-1, -1):
            layer_k = network.layers[k]
            network.layers[k].weight_matrix.matrix -= (layer_k.gradient_matrix.matrix * alpha)
            logger.info('\n\tTheta{} = \n{}'.format(k+1, network.layers[k].weight_matrix.str_tabs(2)))
        
        logger.info('\n')



if __name__ == "__main__":
    main()
