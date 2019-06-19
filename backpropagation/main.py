import logging
import click
import click_log
import backpropagation as bp

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

cli = click.Group()

@cli.command(name='backprop')
@click_log.simple_verbosity_option(logger)
@click.argument('network_filename')
@click.argument('initial_weights_filename')
@click.argument('data_set_filename')
@click.option('--alpha', '-a', default=0.5, help='Weights Update Rate, is used to smooth the gradient')
@click.option('--max_iterations', '-it', default=100, help='Maximum number of iterations.')
def backprop(network_filename, initial_weights_filename, data_set_filename,
             max_iterations, alpha):

    bp.logger = logger
    bp.backpropagation(network_filename, initial_weights_filename, data_set_filename,
                    max_iterations, alpha)



if __name__ == '__main__':
    cli()

