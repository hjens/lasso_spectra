Package overview
===================
`lasso_spectra` is a package for fitting Lasso regression models to data,
specifically galaxy spectra. It contains two different classes for performing
the actual model fitting. `GeneralizedLasso` is a `tensorflow` implementation
of Lasso regression, which includes the ability to use link functions.
`SKLasso` is a wrapper around the `scikit-learn` Lasso implementation intended
to give the same syntax as `GeneralizedLasso`. It is much faster and more
reliable, but does not support generalized linear models. Use `SKLasso` if
unsure.

Package installation
====================
Download the package and put it in your python path. Make sure you can
import it:
```
import lasso_spectra
```

You need to have `numpy` and `scipy` installed. For `GeneralizedLasso` you
also need `tensorflow`, and for `SKLasso` you need `scikit-learn`.