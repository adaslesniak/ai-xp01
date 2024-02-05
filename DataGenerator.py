matrix_size = 5
data_file_name = 'matrices'
"""
Generate two csv files that conatins data for training 
and data for tests. Each csv will contain rows of numbers that are flattened matrixes 
and additional row that contains true/false information if given matrix has 
determinant different than zero.
All matrices have values in range of 0-1 exclusive of 1.
"""

import numpy as np
import pandas as pd
matrices_hashes = set()


def _matrix_hash(matrix):
    return tuple(np.round(matrix, decimals=4).flatten())


def _generate_unique_matrix(singular):
    while True:
        another_matrix = _generate_matrix(singular)
        m_hash = _matrix_hash(another_matrix)
        if m_hash not in matrices_hashes:
            matrices_hashes.add(m_hash)
            return another_matrix

def _generate_matrix(singular=False):
    r_matrix = np.random.rand(5, 5)
    if not singular:
        return r_matrix
    else:
        row_to_duplicate = np.random.randint(0, 5)
        r_matrix[row_to_duplicate] = r_matrix[(row_to_duplicate + 1) % 5]  # Duplicate a row
        return r_matrix
    

def _genarate_data_set(size):
    ds = []
    for i in range(size): 
        next = _generate_unique_matrix(i % 3 == 0)
        determinant = np.linalg.det(next)
        label = 1 if determinant != 0 else 0
        next_datum = next.flatten().tolist() + [label]
        ds.append(next_datum)
    np.random.shuffle(ds)
    return ds


def _write_file(data, set_name):
    df = pd.DataFrame(data, columns=[f'm{i}' for i in range(1, 26)] + ['label'])
    df.to_csv(f'{data_file_name}_{set_name}.csv', index=False)


def generate_data(training_data_size = 33, test_data_size = 6):
    matrices_hashes.clear()
    training_data = _genarate_data_set(training_data_size * 1000)
    _write_file(training_data, 'training')
    test_data = _genarate_data_set(test_data_size * 1000)
    _write_file(test_data, 'test')
    print('Data generated')


def _load_data(set_name):
    data_file = f'{data_file_name}_{set_name}.csv'
    file_content = pd.read_csv(data_file)
    data = file_content.drop('label', axis=1).values
    labels = file_content['label'].values
    print(f'file: {file_content.shape} - data: {data.shape} - labels: {labels.shape}')
    return data, labels


def load_training_data():
    return _load_data('training')


def load_test_data():
    return _load_data('test')


generate_data()
