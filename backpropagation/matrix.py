#import numpy

class Matrix:
    def __init__(self):
        self.matrix = []
        self.num_rows = 0
        self.num_cols = 0

    def add_row(self, row):
        numElem = len(row)
        if self.num_cols>0 and numElem != self.num_cols:
            self.print()
            print('row = {}'.format(row))
            raise Exception('Trying add row with size={} at matrix with {} cols.'.format(numElem, self.num_cols))
        
        self.matrix.append(row)
        self.num_rows += 1
        self.num_cols = numElem
    

    def getElem(self, row, col):
        return self.matrix[row][col]


    def multiply_by_vector(self, vector):
        if len(vector) != self.num_cols:
            self.print()
            print(vector)
            raise Exception('Vector dimension is {}, mas matrix has {} cols'.format(len(vector), self.num_cols))
        
        result = []

        for r in range(0, self.num_rows):
            product = 0
            for c in range(0, self.num_cols):
                product += self.getElem(r,c) * vector[c]
            result.append(product)

        return result


    def print(self):
        print('Matrix %i x %i' %(self.num_rows, self.num_cols))
        for r in range(0, self.num_rows):
            print('[', end=' ')
            for c in range(0, self.num_cols-1):
                print(self.matrix[r][c], end='\t')
            c = self.num_cols-1
            print(self.matrix[r][c], end=' ]\n')

''' ------------------------------------------------- '''

def dot_product(array1, array2):
    n1 = len(array1)
    
    if n1 != len(array2):
        print('a1 = {}'.format(array1))
        print('a2 = {}'.format(array2))
        raise Exception('dot_product receve arrays with different lenghts.')
    
    result = 0
    for i in range(0, n1):
        result += array1[i] * array2[i]
    return result
    



    
