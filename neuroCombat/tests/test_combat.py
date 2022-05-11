import numpy as np
import pandas as pd

from ..combat import Combat
from ..neuroCombat import neuroCombat, neuroCombatFromTraining

def gen_fake_data(size=100, n_data_feats=3, n_batches=3,
                  n_cont_covars=2, n_cat_covars=1, include_y=True,
                  y_is_cat=False):

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
    if include_y:
        if y_is_cat:
            y = pd.Series(np.random.randint(0, 3, size=size))
            nc_args['categorical_cols'].append('y')
        else:
            y = pd.Series(np.random.random(size))
            nc_args['continuous_cols'].append('y')
    else:
        y = None
        
    # Gen batch series
    batch = pd.Series(np.random.randint(0, n_batches, size=size))
    batch = batch.astype('category')

    nc_args['covars'] = covars.copy()
    nc_args['covars']['batch'] = batch
    nc_args['covars']['y'] = y
    nc_args['batch_col'] = 'batch'
        
    return data, batch, covars, y, nc_args

def test_base_consistency_with_neuroCombat_1():

    data, batch, covars, y, nc_args = gen_fake_data(size=200)

    nc_res = neuroCombat(**nc_args, eb=True, parametric=True)
    combat = Combat(batch=batch, covars=covars, include_y=True, eb=True, parametric=True)
    X_trans = combat.fit_transform(data, y)

    # Combat class based version just varies w.r.t. to some dtype, specifically,
    # it doesn't ever explicitly force a lower resolution
    assert np.allclose(nc_res['data'].T, X_trans)

def test_base_consistency_with_neuroCombat_2():
    
    # Try with parametric False
    data, batch, covars, y, nc_args = gen_fake_data(size=200, n_batches=2)

    nc_res = neuroCombat(**nc_args, eb=True, parametric=False)
    combat = Combat(batch=batch, covars=covars, include_y=True, eb=True, parametric=False)
    X_trans = combat.fit_transform(data, y)

    # Combat class based version just varies w.r.t. to some dtype, specifically,
    # it doesn't ever explicitly force a lower resolution
    assert np.allclose(nc_res['data'].T, X_trans)

def test_base_consistency_with_neuroCombat_3():
    
    # Try with eb, False
    data, batch, covars, y, nc_args = gen_fake_data(size=200, n_cat_covars=0)

    nc_res = neuroCombat(**nc_args, eb=False, parametric=True)
    combat = Combat(batch=batch, covars=covars, include_y=True, eb=False, parametric=True)
    X_trans = combat.fit_transform(data, y)

    # Combat class based version just varies w.r.t. to some dtype, specifically,
    # it doesn't ever explicitly force a lower resolution
    assert np.allclose(nc_res['data'].T, X_trans)

def test_consistency_with_neuroCombat_fit_from_train():

    # For matched version w/ fit from train, don't use y
    data, batch, covars, _, nc_args = gen_fake_data(size=200, include_y=False)

    # Fit each at first
    nc_res = neuroCombat(**nc_args, eb=True, parametric=True)

    # Make sure with lining up with neuroCombat from training,
    # we pass parameter use_mean_for_ne
    combat = Combat(batch=batch, covars=covars, include_y=False, eb=True,
                    parametric=True, use_mean_for_new=True)
    X_trans = combat.fit_transform(data)

    # Just base check
    assert np.allclose(nc_res['data'].T, X_trans)

    # Now we want to make sure results match when using fit from train, on just the same samples
    from_train_res = neuroCombatFromTraining(dat=nc_args['dat'], batch=np.array(batch), estimates=nc_res['estimates'])
    X_trans2 = combat.transform(data)
    assert np.allclose(from_train_res['data'].T, X_trans2)

    # Note that even though this is the same data, the results are not consistent
    # Since we use mean for new basically, even without y data
    assert not np.allclose(from_train_res['data'], nc_res['data'])

    # This is not the case if we use Combat class, with use_mean_for_new off.
    # This results will be identical (unless we are using include_y)
    combat = Combat(batch=batch, covars=covars, include_y=False, eb=True,
                    parametric=True, use_mean_for_new=False)
    X_trans = combat.fit_transform(data)
    X_trans2 = combat.transform(data)
    assert np.array_equal(X_trans, X_trans2)

def test_consistency_interesting_case():

    # So one other thing we can look at with Combat vs. neuroCombat method
    # is what happens when we are using a y / target variable

    # Setup data with just 1 cont covariate, which we will use as y
    data, batch, covars, _, nc_args = gen_fake_data(size=200, n_data_feats=3, n_batches=3,
                                                    n_cont_covars=1, n_cat_covars=0,
                                                    include_y=False)

    # So next we fit the base neurocombat, just pretending cont. feat 1 is y
    # neuroCombatFromTraining doesn't touch covar's so this isn't a big deal
    nc_res = neuroCombat(**nc_args, eb=True, parametric=True)

    # For our combat object though, we pass covars as None, and will use it instead for y
    combat = Combat(batch=batch, covars=None, include_y=True, y_is_cat=False, eb=True,
                    parametric=True, use_mean_for_new=False)
    X_trans = combat.fit_transform(data, covars['covars_0'])

    # So first, let's just confirm that the fit behavior matches
    # as in the other tests
    assert np.allclose(nc_res['data'].T, X_trans)

    # What we are actually interested in showing though is that in predicting in new
    # samples, when we set y internally as the mean from the training, we get the same
    # result as the base using the beta weight mean behavior as fromTraining does, even though
    # the internals vary
    from_train_res = neuroCombatFromTraining(dat=nc_args['dat'], batch=np.array(batch), estimates=nc_res['estimates'])
    X_trans2 = combat.transform(data)
    assert np.allclose(from_train_res['data'].T, X_trans2)

    # This same behavior holds if y is multi-class too, as we set it to the mean after dummy coding
    # Below we just repeat everything but with n_cat_covars=1 and y_is_cat=True
    data, batch, covars, _, nc_args = gen_fake_data(size=200, n_data_feats=3, n_batches=3,
                                                    n_cont_covars=0, n_cat_covars=1, include_y=False)
    nc_res = neuroCombat(**nc_args, eb=True, parametric=True)
    combat = Combat(batch=batch, covars=None, include_y=True, y_is_cat=True, eb=True, parametric=True, use_mean_for_new=False)
    X_trans = combat.fit_transform(data, covars['covars_0'])
    assert np.allclose(nc_res['data'].T, X_trans)
    from_train_res = neuroCombatFromTraining(dat=nc_args['dat'], batch=np.array(batch), estimates=nc_res['estimates'])
    X_trans2 = combat.transform(data)
    assert np.allclose(from_train_res['data'].T, X_trans2)

def test_batch_as_str():

    size = 200
    n_data_feats = 3
    data, batch, covars, _, _ = gen_fake_data(size=size, n_data_feats=n_data_feats)

    covars['batch'] = batch
    combat = Combat(batch='batch', covars=covars, include_y=False, eb=True,
                    parametric=True)
    data_trans = combat.fit_transform(data)
    assert data_trans.shape == (size, n_data_feats)

def test_no_covars():

    size = 200
    n_data_feats = 3
    data, batch, _, _, _ = gen_fake_data(size=size, n_data_feats=n_data_feats)

    combat = Combat(batch=batch, covars=None, include_y=False, eb=True,
                    parametric=True)
    data_trans = combat.fit_transform(data)
    assert data_trans.shape == (size, n_data_feats)

def test_no_fail_no_y():

    size = 200
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

    size = 200
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

    size = 200
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