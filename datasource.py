import numpy as np
from numpy.lib.recfunctions import append_fields
from distutils.version import LooseVersion


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

    def _shuffle_data(self):
        """ Shuffle data and targets """
        indices = np.arange(self.num_objs)
        np.random.shuffle(indices)
        self.data = self.data[indices, :]
        self.target = self.target[indices]

    def _split_data(self, training_fraction=0.8):
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


class SpectrumTableWide(DataSource):
    """ Data source from a wide spectrum file, where
    each feature has its own column
    """

    def __init__(self, filename, target_name='fesc', exclude_cols=[],
                 training_fraction=0.8):
        """
        Initialize the data source with a file.
        Also shuffle the data and divide into a test set and
        a training set.

        :param filename: The file name to read
        :param target_name: The name of the target column
        :param exclude_cols: List of names of columns to exclude
        :param training_fraction: The fraction of the data to
        use for training
        """
        # Read the data. We pass the weird string to deletechars to
        # keep genfromtxt from removing periods in the column names
        # This does not seem to work in older versions of numpy,
        # so check this first.
        if LooseVersion(np.version.version) < LooseVersion('1.10.4'):
            print 'Warning! numpy version ', np.version.version, \
            'may not read column names with periods properly. Check your' \
            'feature_id property, and consider updating numpy.'
        self.raw_data = np.genfromtxt(filename, names=True,
                                      delimiter=',',
                                      deletechars="""~!@#$%^&*()=+~\|]}[{';: /?>,<""")
        self._make_design_matrix(target_col=target_name,
                                 exclude_cols=exclude_cols)
        self._shuffle_data()
        self._split_data(training_fraction)

    def _make_design_matrix(self, target_col, exclude_cols):
        """
        Extract the design matrix and target vector from the
        raw data

        :param target_col: The name of the column containin the target
        :param exclude_cols: A list of columns that we want to exclude
        """
        # Check type of exclude_cols
        if not hasattr(exclude_cols, '__iter__'):
            raise ValueError('exclude_cols must be a list of column names, or '
                             'an empty list')
        # We always want to exclude the target column from the design matrix
        if not target_col in exclude_cols:
            exclude_cols.append(target_col)
        # Target vector
        self.target = self.raw_data[target_col]
        # Copy full data and remove unused columns
        self.data = self.raw_data.copy()
        self.data = self._rmfield(self.data, exclude_cols)
        self.num_objs = len(self.data)
        self.num_features = len(self.data.dtype.names)
        # Extract feature names
        self.feature_id = map(float, self.data.dtype.names)
        # Convert to numpy matrix
        self.data = self.data.view((float, len(self.data.dtype.names)))

    def _rmfield(self, a, fieldnames_to_remove):
        """ Utility function to remove named columns """
        return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]


class SpectrumTableNarrow(DataSource):
    """ Data source from a narrow spectrum file, where
    each variable has its own column """

    def __init__(self, filename, target_name='fesc', feature_name='flux_noisy',
                 feature_id='wavel', training_fraction=0.8):
        """ Initialize the data source with a file.
        Also shuffle the data and divide into a test set
        and a training set.

        :param filename: The file name to read
        :param target_name: The name of the target column
        :param feature_name: The name of the features column
        :param feature_id: The column containing the name or
        value identifying each feature
        :param training_fraction: The fraction of the data to
        use for training
        """
        self.raw_data = np.genfromtxt(filename, names=True,
                                  delimiter=',')

        self._make_design_matrix(target_col=target_name, feature_col=feature_name,
                                 id_name1='gal_id', id_name2=target_name,
                                 feature_id=feature_id)
        self._shuffle_data()
        self._split_data(training_fraction)

    def _make_design_matrix(self, target_col, feature_col,
                            id_name1, id_name2, feature_id='wavel'):
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
