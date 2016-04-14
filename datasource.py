import numpy as np
from numpy.lib.recfunctions import append_fields


class DataSource:
    """ A source of data for the machine learning routines
    """

    data = None
    target = None
    training_data = None
    training_target = None
    test_data = None
    test_target = None
    num_features = 0
    num_objs = 0
    feature_id = None

    def __init__(self):
        pass

    def shuffle_data(self):
        """ Shuffle data and targets """
        indices = np.arange(self.num_objs)
        np.random.shuffle(indices)
        self.data = self.data[indices, :]
        self.target = self.target[indices]

    def split_data(self, training_fraction=0.8):
        """ Split data into a training set and a test set

        :param training_fraction: The fraction of the data to use
        for training
        """
        if training_fraction < 0. or training_fraction > 1.0:
            raise ValueError('training_fraction must be between 0 and 1')
        num_train_objs = int(self.num_objs*training_fraction)
        self.training_data = self.data[:num_train_objs,:]
        self.training_target = self.target[:num_train_objs]
        self.test_data = self.data[num_train_objs:,:]
        self.test_target = self.target[num_train_objs:]


class NarrowSpectraTable (DataSource):
    """ Data source from a narrow spectra file, where
    each variable has its own column """

    def __init__(self, filename, target_name='fesc', feature_name='flux_noisy',
                 feature_id='wavel'):
        """ Initialize the data source with a file.
        Also shuffle the data and divide into a test set
        and a training set.

        :param filename: The file name to read
        :param target_name: The name of the target column
        :param feature_name: The name of the features column
        :param feature_id: The column containing the name or
        value identifying each feature
        """
        self.raw_data = np.genfromtxt(filename, names=True,
                                  delimiter=',')
        self.colnames = self.raw_data.dtype.names

        self._make_design_matrix(target_col=target_name, feature_col=feature_name,
                                 id_name1='gal_id', id_name2=target_name,
                                 feature_id=feature_id)
        self.shuffle_data()
        self.split_data()

    def _make_design_matrix(self, target_col, feature_col,
                            id_name1='gal_id', id_name2='fesc',
                            feature_id='wavel'):
        """ Create the design matrix from the raw data

        :param target_col: The name of the target column
        :param feature_col: The name of the feature column
        :param id_name1: The name of the first id column
        :param id_name2: The name of the second id column
        :param feature_id: The column containing the name or
        value identifying each feature
        """

        # Figure out how many unique objects there are
        col1 = self.raw_data[id_name1]
        col2 = self.raw_data[id_name2]
        first_obj = col1[(col1 == col1[0]) * (col2 == col2[0])]
        self.num_features = len(first_obj)
        self.num_objs = len(np.unique(col1)) * len(np.unique(col2))

        # Make unique index
        unique_id = np.repeat(np.arange(self.num_objs), self.num_features)
        self.raw_data = append_fields(self.raw_data, 'unique_id', unique_id)

        # Create design matrix and target vector
        self.data = np.zeros((self.num_objs, self.num_features))
        self.target = np.zeros(self.num_objs)
        for i in range(self.num_objs):
            idx = self.raw_data['unique_id'] == i
            self.data[i, :] = self.raw_data[feature_col][idx]
            self.target[i] = self.raw_data[target_col][idx][0]

        if not feature_id is None:
            self.feature_id = self.raw_data[feature_id][idx]