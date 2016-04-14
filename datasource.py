import numpy as np

class DataSource:
    def __init__(self, filename, **kwargs):
        '''
        Initialize a data source with a csv file

        Parameters:
            filename: string
             The file to read
        '''
        delimiter = kwargs.get('delimiter', ',')
        data = np.genfromtxt(filename, **kwargs)