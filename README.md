# Multi-site harmonization in Python with neuroCombat

[![License: MIT](https://img.shields.io/github/license/Jfortin1/neuroCombat)](https://opensource.org/licenses/MIT) 
[![Version](https://img.shields.io/pypi/v/neuroCombat)](https://pypi.org/project/neuroCombat/)
[![PythonVersion](https://img.shields.io/pypi/pyversions/neuroCombat)]()


This is the maintained and official version of neuroCombat (previously hosted [here](https://github.com/ncullen93/neuroCombat)) introduced in our [our recent paper](https://www.sciencedirect.com/science/article/pii/S105381191730931X).


## Installation

neuroCombat is hosted on PyPI, and the easiest way to install neuroCombat is to use the ```pip``` command:

```
pip install neuroCombat
```

## Usage

The ```neuroCombat``` function performs harmonization 

```python
from neuroCombat import neuroCombat
import pandas as pd
import numpy as np

# Getting example data
# 200 rows (features) and 10 columns (scans)
data = np.genfromtxt('testdata/testdata.csv', delimiter=",", skip_header=1)

# Specifying the batch (scanner variable) as well as a biological covariate to preserve:
covars = {'batch':[1,1,1,1,1,2,2,2,2,2],
          'gender':[1,2,1,2,1,2,1,2,1,2]} 
covars = pd.DataFrame(covars)  

# To specify names of the variables that are categorical:
categorical_cols = ['gender']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'batch'

#Harmonization step:
data_combat = neuroCombat(dat=data,
    covars=covars,
    batch_col=batch_col,
    categorical_cols=categorical_cols)["data"]
```

## Optional arguments

- `eb` : `True` or `False`. Should Empirical Bayes be performed? If `False`, the harmonization model will be fit for each feature separately. This is equivalent to performing a location/shift (L/S) correction to each feature separately (no information pooling across features). 

- `parametric` : `True` or `False`. Should parametric adjustements be performed? `True` by default. 

- `mean_only` : `True` or `False`. Should only be means adjusted (no scaling)? `False` by default

- `ref_batch` : batch name to be used as the reference batch for harmonization. `None` by default, in which case the average across scans/images/sites is taken as the reference batch.

## Output

Since version 0.2.10, the `neuroCombat` function outputs a dictionary with 3 elements:
- `data`: A numpy array of the harmonized data, with the same dimension (shape) as the input data.
- `estimates`: A dictionary of the neuroCombat estimates; useful for visualization and understand scanner effects.
- `info`: A dictionary of the inputs needed for ComBat harmonization (batch/scanner information, etc.)

To simply return the harmonized data, one can use the following:

```
data_combat = neuroCombat(dat=dat, ...)["data"]
```

where `...` are the user-specified arguments needed for harmonization. 
    

