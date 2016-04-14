import numpy as np
from numpy.lib.recfunctions import append_fields


class DataSource:
    design_matrix = None
    target_vector = None
    num_features = 0
    num_objs = 0

    def __init__(self):
        pass


class NarrowSpectraTable (DataSource):

    def __init__(self, filename, target_name='fesc', feature_name='flux_noisy'):
        '''
        Initialize a data source with a csv file

        Parameters:
            filename: string
             The file to read
        '''
        self.data = np.genfromtxt(filename, names=True,
                                  delimiter=',')
        self.colnames = self.data.dtype.names

        self._make_design_matrix(target_col=target_name, feature_col=feature_name,
                                 id_name1='gal_id', id_name2=target_name)


    def _make_design_matrix(self, target_col, feature_col,
                            id_name1='gal_id', id_name2='fesc'):
        '''
        Generate a design matrix

        :param target_col:
        :param feature_col:
        :return:
        '''
        # Figure out how many unique objects there are
        col1 = self.data[id_name1]
        col2 = self.data[id_name2]
        first_obj = col1[(col1 == col1[0]) * (col2 == col2[0])]
        self.num_features = len(first_obj)
        self.num_objs = len(np.unique(col1)) * len(np.unique(col2))

        # Make unique index
        unique_id = np.repeat(np.arange(self.num_objs), self.num_features)
        self.data = append_fields(self.data, 'unique_id', unique_id)

        # Create design matrix and target vector
        self.design_matrix = np.zeros((self.num_objs, self.num_features))
        self.target_vector = np.zeros(self.num_objs)
        for i in range(self.num_objs):
            idx = self.data['unique_id'] == i
            self.design_matrix[i,:] = self.data[feature_col][idx]
            self.target_vector[i] = self.data[target_col][idx][0]