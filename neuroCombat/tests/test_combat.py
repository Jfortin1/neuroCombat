import numpy as np
import pandas as pd

from ..combat import Combat
from ..neuroCombat import neuroCombat, neuroCombatFromTraining

def gen_fake_data(size=100, n_data_feats=3, n_batches=3,
                  n_cont_covars=2, n_cat_covars=1, y_is_cat=False):

    nc_args = {}
    
    # Fill covars df
    covars = pd.DataFrame()

    nc_args['continuous_cols'] = []
    for i in range(n_cont_covars):
        covars[f'covars_{i}'] = np.random.random(size)
        nc_args['continuous_cols'].append(f'covars_{i}')

    nc_args['categorical_cols'] = []
    for i in range(n_cont_covars, n_cat_covars + n_cont_covars):
        # Default number of categories, idea is multi-class is more general
        covars[f'covars_{i}'] = np.random.randint(0, 3, size=size)
        covars[f'covars_{i}'] = covars[f'covars_{i}'].astype('category')
        nc_args['categorical_cols'].append(f'covars_{i}')
        
    if n_cont_covars + n_cat_covars == 0:
        covars = None
        
    # Gen data
    data = pd.DataFrame()
    for i in range(n_data_feats):
        data[str(i)] = np.random.random(size)
    data = data.astype('float32')

    nc_args['dat'] = np.array(data).T

    # Gen y
    if y_is_cat:
        y = pd.Series(np.random.randint(0, 3, size=size))
        nc_args['categorical_cols'].append('y')
    else:
        y = pd.Series(np.random.random(size))
        nc_args['continuous_cols'].append('y')
        
    # Gen batch series
    batch = pd.Series(np.random.randint(0, n_batches, size=size))
    batch = batch.astype('category')

    nc_args['covars'] = covars.copy()
    nc_args['covars']['batch'] = batch
    nc_args['covars']['y'] = y
    nc_args['batch_col'] = 'batch'
        
    return data, batch, covars, y, nc_args


def test_consistency_with_neuroCombat1():

    size = 150
    n_data_feats = 3
    data, batch, covars, y, nc_args = gen_fake_data(size=size, n_data_feats=n_data_feats)

    nc_res = neuroCombat(**nc_args, eb=True, parametric=True)
    combat = Combat(batch=batch, covars=covars, include_y=False, eb=True, parametric=True)
    X_trans = combat.fit_transform(data, y)

    assert np.allclose(nc_res['data'].T[0], X_trans[0])

def test_batch_as_str():

    size = 150
    n_data_feats = 3
    data, batch, covars, _, _ = gen_fake_data(size=size, n_data_feats=n_data_feats)

    covars['batch'] = batch
    combat = Combat(batch='batch', covars=covars, include_y=False, eb=True,
                    parametric=True)
    data_trans = combat.fit_transform(data)
    assert data_trans.shape == (size, n_data_feats)

def test_no_covars():

    size = 150
    n_data_feats = 3
    data, batch, _, _, _ = gen_fake_data(size=size, n_data_feats=n_data_feats)

    combat = Combat(batch=batch, covars=None, include_y=False, eb=True,
                    parametric=True)
    data_trans = combat.fit_transform(data)
    assert data_trans.shape == (size, n_data_feats)

def test_no_fail_no_y():

    size = 150
    n_data_feats = 3
    data, batch, covars, _, _ = gen_fake_data(size=size, n_data_feats=n_data_feats)

    combat = Combat(batch=batch, covars=covars, include_y=False, eb=True,
                    parametric=True)
    data_trans = combat.fit_transform(data)
    assert data_trans.shape == (size, n_data_feats)

    # Re-fitting on 'new' / the exact same data should give same result
    data_trans2 = combat.fit_transform(data)

    assert np.array_equal(data_trans, data_trans2)

def test_no_fail_y():

    size = 150
    n_data_feats = 5
    data, batch, covars, y, _ = gen_fake_data(size=size, n_data_feats=n_data_feats)

    combat = Combat(batch=batch, covars=covars, include_y=True, eb=True,
                    parametric=True)
    data_trans = combat.fit_transform(data, y)
    assert data_trans.shape == (size, n_data_feats)

    # W/ Y should just be kind of close
    data_trans2 = combat.fit_transform(data)
    assert np.allclose(data_trans, data_trans2, atol=1e3, rtol=0.001)

def test_no_fail_y_cat():

    size = 150
    n_data_feats = 2
    data, batch, covars, y, _ = gen_fake_data(size=size, n_data_feats=n_data_feats,
                                              y_is_cat=True)

    combat = Combat(batch=batch, covars=covars, include_y=True,
                    y_is_cat=True, eb=True,
                    parametric=True)
    data_trans = combat.fit_transform(data, y)
    assert data_trans.shape == (size, n_data_feats)

    # W/ Y should just be kind of close
    data_trans2 = combat.fit_transform(data)
    assert np.allclose(data_trans, data_trans2, atol=1e3, rtol=0.001)