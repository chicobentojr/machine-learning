import pandas as pd


if __name__ == "__main__":
    dataset = pd.read_csv('datasets/benchmark.csv', sep=';')

    print(dataset)
    print('\nSummary')
    print(dataset.describe())
