# Imports. Change path names as required
import sys
sys.path.append('..')
import lasso_spectra
import pylab as pl
import numpy as np
import pickle
import os

np.random.seed(4)

# Read data. The package comes with two csv files containing data in
# different formats

# Use this class to read a narrow table with one column per variable
# you need to specify the name of the column containing the property
# to predict (the target) and the column containing the values of the
# features (usually the flux of the spectra)
#datasource = lasso_spectra.SpectrumTableNarrow('FiGeLMC_R100_sn5.csv',
#                                              target_name='fesc',
#                                              feature_name='flux_noisy')

# Use this class if the data is in a format with one galaxy per row.
# Here, we also specify the target name as above. We can also give a
# list of columns to exclude. In this case, the file contains magnitudes,
# which we do not want to use in the prediction.
datasource = lasso_spectra.SpectrumTableWide('FiGeLMC_R100_sn5_wide.csv',
                                             target_name='fesc',
                                             exclude_cols=['m_ab'])

# Fit a model to the data. The function fit_CV takes the training data
# set (spectra) and the targets (escape fractions) finds the best model
# using cross-validation. The regularization parameter (alpha), is varied
# to find a model that does not overfit the data.
lasso_model = lasso_spectra.SKLasso()
alphas = 10**np.linspace(-3,-1,100)
lasso_model.fit_CV(datasource.training_data, datasource.training_target,
                   n_folds=10, alphas=alphas)

# Now that we have a model, we can use it to predict escape fractions
# from new spectra. By default, the data source object has already
# split the data set into a training set and a test set. We can evaluate
# the performance on the test set using the predict function.
true_fesc = datasource.test_target
predicted_fesc = lasso_model.predict(datasource.test_data)

# The model can be saved and loaded later if we don't want to retrain
# it.
tempfile = 'lasso_temp.bin'
with open(tempfile, 'w') as f:
    pickle.dump(lasso_model, f)
with open(tempfile, 'r') as f:
    lasso_model = pickle.load(f)
os.remove(tempfile)

# Plot true vs predicted fesc
pl.subplot(131)
pl.plot(true_fesc, predicted_fesc, 'o')
pl.xlabel('True fesc')
pl.ylabel('Predicted fesc')

# Plot the model coefficients for each wavelength. The predicted
# escape fraction is the sum of these coefficients multiplied with
# the flux in the respective bins, plus a constant bias term.
pl.subplot(132)
pl.plot(datasource.feature_id, lasso_model.coeffs, 'o-')
pl.xlabel('Wavelength')
pl.ylabel('Coefficient value')

# Plot the mean-squared cross-validation error as a function of the
# regularization parameter, alpha. The range of alpha, should be chosen
# so that there is a minimum in this function.
pl.subplot(133)
pl.loglog(alphas, lasso_model.alpha_mse, '-')
pl.xlabel('Regularization parameter')
pl.ylabel('MSE')
pl.show()