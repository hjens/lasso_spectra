import sys
sys.path.append('..')
import lasso_spectra
import pylab as pl
import numpy as np

np.random.seed(4)

# Read data
datasource = lasso_spectra.NarrowSpectraTable('FiGeLMC_R100_sn5.csv')

# Fit a model
lasso_model = lasso_spectra.SKLasso()
lasso_model.fit_CV(datasource.training_data, datasource.training_target,
                   n_folds=10, alphas=10**np.linspace(-3,1,100))
print lasso_model.alpha

# Predict results for test set
true_fesc = datasource.test_target
predicted_fesc = lasso_model.predict(datasource.test_data)

# Plot true vs predicted fesc, and model coefficients
pl.subplot(121)
pl.plot(true_fesc, predicted_fesc, 'o')
pl.xlabel('True fesc')
pl.ylabel('Predicted fesc')
pl.subplot(122)
pl.plot(datasource.feature_id, lasso_model.coeffs, 'o-')
pl.xlabel('Wavelength')
pl.ylabel('Coefficient value')
pl.show()