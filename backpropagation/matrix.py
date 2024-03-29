import numpy as np

np.set_printoptions(precision=5)

def str_tabs(numpy_matrix, num_tabs):
    tabs = '\t' * num_tabs
    output = tabs + '{}'.format(numpy_matrix)
    return output.replace('\n', '\n'+tabs)


class Matrix:
    def __init__(self, rows=[]):
        self.matrix = np.matrix(rows)
        self.num_rows = len(rows)
        self.num_cols = (0 if self.num_rows==0 else len(rows[0]))

    def set(self, num_rows : int, num_cols : int):
        self.matrix = np.zeros((num_rows, num_cols))
        self.num_rows = num_rows
        self.num_cols = num_cols

    def set_from_numpy(self, numpy_matrix):
        self.matrix = numpy_matrix.copy()
        (self.num_rows, self.num_cols) = numpy_matrix.shape

    def copy(self):
        return Matrix(self.matrix.tolist())


    def transpose_and_multiply_by_vector(self, vector):
        matT = self.matrix.copy().transpose()
        return np.matmul(matT, vector).tolist()[0]

    
    def multiply_by_vector(self, vector):
        if len(vector) != self.num_cols:
            print(self.matrix)
            print(vector)
            raise Exception('Vector dimension is {}, mas matrix has {} cols'.format(len(vector), self.num_cols))

        return np.matmul(self.matrix, vector).tolist()[0]        


    def sum_square_elements(self):
        func = np.vectorize(lambda x: x ** 2)
        mat = func(self.matrix.copy())
        return np.sum(mat)

    def sum_square_weights_without_bias(self):
        func = np.vectorize(lambda x: x ** 2)
        mat = func(self.matrix.copy())
        total = np.sum(mat)
        bias_col = mat[:,0]
        total_bias = np.sum(bias_col)
        return total - total_bias

    def regularize(self, regularizationFactor): # set bias column to zero
        self.matrix[:,0] = np.zeros(shape = (self.num_rows, 1))
        self.matrix *= regularizationFactor
        

    def __str__(self):
        return '{}'.format(self.matrix)

    def str_tabs(self, num_tabs):
        return str_tabs(self.matrix, num_tabs)

    def print(self, name=''):
        print('\nMatrix ({} x {}):\n{} =\n{}\n'.format(self.num_rows, self.num_cols, name, self.matrix))
    
#--------------------------------------

def numpy_to_Matrix(numpy_matrix):
    M = Matrix()
    M.set_from_numpy(numpy_matrix)
    return M

