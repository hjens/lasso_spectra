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
also need `tensorflow` (https://www.tensorflow.org), and for `SKLasso`
you need `scikit-learn`.

Usage
=====
The file `example.py` shows an example of how to use the package.

In short, there are three steps to using `lasso_spectra`:

1. Load your data
2. Fit a model
3. Use the model to predict values from new data

The first step can be done quickly using the classes `SpectrumTableNarrow` and
`SpectrumTableWide`. These classes read csv files containing galaxy spectra and
convert them into the format required for the model fitting. Use `SpectrumTableNarrow`
if your data is in a narrow format, with one column per variable. Use `SpectrumTableWide`
if your data contains one column for each wavelength bin (or other feature).

If you have data in other formats, you can subclass the class `DataSource`, or
write your own data loading code from scratch. See the documentation for
`SKLasso` to find out the correct values format of the data.

For the second step, you first need to create a model object, either of the class
`SKLasso` or `GeneralizedLasso`. Then, you typically use the function `fit_CV`, passing
in the training data and targets from your data source. This function will try many
different values of the regularization parameter (called lambda in Jensen et al 2016, but
referred to as alpha here to keep with the syntax of `scikit-learn`). Make sure to pass
a wide range of alphas to the function to make sure it can find the one that gives the
smallest cross-validation error. After the fitting is done, you may save the model using
pickling.

Finally, you can use the model to predict values of your target (for example fesc) on
new data. This is done using the `predict` function.