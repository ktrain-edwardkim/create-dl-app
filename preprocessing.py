import pandas as pd


def load_dataset():
    # The Auto MPG dataset
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    # Get the data
    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()

    # Check the dataset
    dataset = dataset.dropna()

    # Convert 'Origin' column to categorical data type
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

    # One-hot encode the 'Origin' column
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

    return dataset
