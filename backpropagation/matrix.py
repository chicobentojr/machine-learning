import numpy

numpy.set_printoptions(precision=6)

def str_tabs(numpy_matrix, num_tabs):
    tabs = '\t' * num_tabs
    output = tabs + '{}'.format(numpy_matrix)
    return output.replace('\n', '\n'+tabs)


class Matrix:
    def __init__(self, rows=[]):
        self.matrix = numpy.matrix(rows)
        self.num_rows = len(rows)
        self.num_cols = (0 if self.num_rows==0 else len(rows[0]))

    def set(self, num_rows : int, num_cols : int):
        self.matrix = numpy.zeros((num_rows, num_cols))
        self.num_rows = num_rows
        self.num_cols = num_cols


    def transpose_and_multiply_by_vector(self, vector):
        matT = self.matrix.copy().transpose()
        return numpy.matmul(matT, vector).tolist()[0]

    
    def multiply_by_vector(self, vector):
        if len(vector) != self.num_cols:
            print(self.matrix)
            print(vector)
            raise Exception('Vector dimension is {}, mas matrix has {} cols'.format(len(vector), self.num_cols))

        return numpy.matmul(self.matrix, vector).tolist()[0]        


    def sum_square_weights_without_bias(self):
        func = numpy.vectorize(lambda x: x ** 2)
        mat = func(self.matrix.copy())
        total = numpy.sum(mat)
        bias_col = mat[:,0]
        total_bias = numpy.sum(bias_col)
        return total - total_bias

    def __str__(self):
        return '{}'.format(self.matrix)

    def str_tabs(self, num_tabs):
        return str_tabs(self.matrix, num_tabs)

    def print(self):
        print('Matrix %i x %i' %(self.num_rows, self.num_cols))
        print(self.matrix)
    
