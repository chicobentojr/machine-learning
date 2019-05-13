import pandas as pd
import click

NUM_FOLDS = 4

def create_folds(filename, separator=';'):
    dataset = pd.read_csv(filename, sep=separator)
    folds = []
    num_instances = dataset.shape[0]
    num_instances_per_fold = num_instances // NUM_FOLDS
    rest = num_instances % NUM_FOLDS

    print('n=%i, per_fold=%i, r=%i' %(num_instances, num_instances_per_fold, rest))

    i=0
    
    #distribute the rest among the first 'rest' folds, that is, each fold with 'num_instances_per_fold + 1' instances
    for c in range(0, rest): 
        folds.append(dataset[i: i + num_instances_per_fold + 1])
        i += num_instances_per_fold + 1

    #last folds with 'num_instances_per_fold' instances
    for c in range(rest, NUM_FOLDS):
        folds.append(dataset[i: i + num_instances_per_fold])
        i += num_instances_per_fold

    return folds


def apply_cross_validation_iteraton(test_fold, training_dataset):
    print('\n\nTEST FOLD')
    print(test_fold)
    print('\ntraining dataset')
    print(training_dataset)


################################# MAIN ##############################################

@click.command()
@click.argument('filename')
@click.option('--separator', '-s', default=';', help='your custom csv separator (e.g.: , or ;)')

def cross_validation(filename, separator=';'):
    folds = create_folds(filename, separator)

    # first fold
    test_fold = folds[0]
    training_dataset = folds[1]
    for j in range(2, NUM_FOLDS):
        training_dataset = training_dataset.append(folds[j])

    apply_cross_validation(test_fold, training_dataset)

    #rest folds
    for i in range(1, NUM_FOLDS):
        test_fold = folds[i]
        training_dataset = folds[0]

        for j in range(1,i):
            training_dataset = training_dataset.append(folds[j])

        for j in range(i+1, NUM_FOLDS):
            training_dataset = training_dataset.append(folds[j])

        apply_cross_validation_iteraton(test_fold, training_dataset)
        

if __name__ == "__main__":
    cross_validation()
