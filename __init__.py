# Check if tensorflow is imported
import imp
try:
    imp.find_module('tensorflow')
    tf_installed = True
except:
    tf_installed = False
if tf_installed:
    from generalized_lasso import *
else:
    print 'tensorflow does not seem to be installed. Skipping import ' \
    'of GeneralizedLasso'

# Import the other stuff
from lasso import *
from datasource import *