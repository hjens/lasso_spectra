import numpy as np
from numpy.lib.recfunctions import append_fields

class DataSource:
    pass


class NarrowSpectraTable:

    def __init__(self, filename, **kwargs):
        '''
        Initialize a data source with a csv file

        Parameters:
            filename: string
             The file to read
        '''
        delimiter = kwargs.get('delimiter', ',')
        self.data = np.genfromtxt(filename, names=True,
                                  delimiter=delimiter, **kwargs)
        self.colnames = self.data.dtype.names

        self._add_unique_id_col()


    def get_design_matrix(self, target_col, feature_col):
        '''
        Generate a design matrix

        :param target_col:
        :param feature_col:
        :return:
        '''
        design_matrix = np.zeros((self.num_objs, self.num_features))
        for i in range(self.num_objs):
            idx = self.data['unique_id'] == i
            design_matrix[i,:] = self.data[feature_col][idx]

        return design_matrix


    def _add_unique_id_col(self, id_name1='gal_id', id_name2='fesc'):
        '''
        Add a new column to the data, with an integer
        that uniquely identifies the object

        :param id_name1:
        :param id_name2:
        :return:
        '''
        # Figure out how many unique objects there are
        col1 = self.data[id_name1]
        col2 = self.data[id_name2]
        first_obj = col1[(col1==col1[0])*(col2==col2[0])]
        self.num_features = len(first_obj)
        self.num_objs = len(np.unique(col1))*len(np.unique(col2))

        # Make unique index
        unique_id = np.repeat(np.arange(self.num_objs), self.num_features)
        self.data = append_fields(self.data, 'unique_id', unique_id)