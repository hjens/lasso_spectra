# Imports. Change path names as required
import sys
sys.path.append('..')
import lasso_spectra
import pylab as pl
import numpy as np
import pickle
import  os

np.random.seed(4)

# Read data
print 'Reading data...'
#datasource = lasso_spectra.SpectrumTableNarrow('FiGeLMC_R100_sn5.csv',
#                                              target_name='fesc',
#                                              feature_name='flux_noisy')

datasource = lasso_spectra.SpectrumTableWide('FiGeLMC_R100_sn5_wide.csv',
                                             target_name='fesc',
                                             exclude_cols=['m_ab'])

# Fit a model
print 'Fitting model...'
lasso_model = lasso_spectra.SKLasso()
lasso_model.fit_CV(datasource.training_data, datasource.training_target,
                   n_folds=10, alphas=10**np.linspace(-3,1,100))

# Predict results for test set
true_fesc = datasource.test_target
predicted_fesc = lasso_model.predict(datasource.test_data)

# The model can be saved and loaded later
print 'Saving and loading...'
tempfile = 'lasso_temp.bin'
with open(tempfile, 'w') as f:
    pickle.dump(lasso_model, f)
with open(tempfile, 'r') as f:
    lasso_model = pickle.load(f)
os.remove(tempfile)

# Plot true vs predicted fesc, and model coefficients
print 'Plotting results...'
pl.subplot(121)
pl.plot(true_fesc, predicted_fesc, 'o')
pl.xlabel('True fesc')
pl.ylabel('Predicted fesc')
pl.subplot(122)
pl.plot(datasource.feature_id, lasso_model.coeffs, 'o-')
pl.xlabel('Wavelength')
pl.ylabel('Coefficient value')
pl.show()