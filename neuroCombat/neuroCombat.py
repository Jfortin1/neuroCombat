# Originally written by Nick Cullen
# Extended and currently maintained by JP Fortin
from __future__ import absolute_import, print_function
import pandas as pd
import numpy as np
import numpy.linalg as la
import math
import copy

def neuroCombat(dat,
           covars,
           batch_col,
           categorical_cols=None,
           continuous_cols=None,
           eb=True,
           parametric=True,
           mean_only=False,
           ref_batch=None):
    """
    Run ComBat to remove scanner effects in multi-site imaging data

    Arguments
    ---------
    dat : a pandas data frame or numpy array
        - neuroimaging data to correct with shape = (features, samples) e.g. cortical thickness measurements, image voxels, etc

    covars : a pandas data frame w/ shape = (samples, covariates)
        - contains the batch/scanner covariate as well as additional covariates (optional) that should be preserved during harmonization.
        
    batch_col : string
        - indicates batch (scanner) column name in covars (e.g. "scanner")

    categorical_cols : list of strings
        - specifies column names in covars data frame of categorical variables to be preserved during harmonization (e.g. ["sex", "disease"])

    continuous_cols : list of strings
        - indicates column names in covars data frame of continuous variables to be preserved during harmonization (e.g. ["age"])

    eb : should Empirical Bayes be performed?
        - True by default

    parametric : should parametric adjustements be performed?
        - True by default

    mean_only : should only be the mean adjusted (no scaling)?
        - False by default

    ref_batch : batch (site or scanner) to be used as reference for batch adjustment.
        - None by default
        
    Returns
    -------
    A dictionary of length 3:
    - data: A numpy array with the same shape as `dat` which has now been ComBat-harmonized
    - estimates: A dictionary of the ComBat estimates used for harmonization
    - info: A dictionary of the inputs needed for ComBat harmonization
    """
    ##############################
    ### CLEANING UP INPUT DATA ###
    ##############################
    if not isinstance(covars, pd.DataFrame):
        raise ValueError('covars must be pandas dataframe -> try: covars = pandas.DataFrame(covars)')

    if not isinstance(categorical_cols, (list,tuple)):
        if categorical_cols is None:
            categorical_cols = []
        else:
            categorical_cols = [categorical_cols]
    if not isinstance(continuous_cols, (list,tuple)):
        if continuous_cols is None:
            continuous_cols = []
        else:
            continuous_cols = [continuous_cols]

    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 

    if isinstance(dat, pd.DataFrame):
        dat = np.array(dat, dtype='float32')



    ##############################

    # get column indices for relevant variables
    batch_col = np.where(covar_labels==batch_col)[0][0]
    cat_cols = [np.where(covar_labels==c_var)[0][0] for c_var in categorical_cols]
    num_cols = [np.where(covar_labels==n_var)[0][0] for n_var in continuous_cols]

    # convert batch col to integer
    if ref_batch is None:
        ref_level=None
    else:
        ref_indices = np.argwhere(covars[:, batch_col] == ref_batch).squeeze()
        if ref_indices.shape[0]==0:
            ref_level=None
            ref_batch=None
            print('[neuroCombat] batch.ref not found. Setting to None.')
            covars[:,batch_col] = np.unique(covars[:,batch_col],return_inverse=True)[-1]
        else:
            covars[:,batch_col] = np.unique(covars[:,batch_col],return_inverse=True)[-1]
            ref_level = covars[np.int(ref_indices[0]),batch_col]
    # create dictionary that stores batch info
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    

     # create design matrix
    print('[neuroCombat] Creating design matrix')
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols, ref_level)
    

    info_dict = {
        'batch_levels': batch_levels,
        'ref_level': ref_level,
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels],
        'design': design
    }

    

   
    # standardize data across features
    print('[neuroCombat] Standardizing data across features')
    s_data, s_mean, v_pool, mod_mean = standardize_across_features(dat, design, info_dict)
    
    # fit L/S models and find priors
    print('[neuroCombat] Fitting L/S model and finding priors')
    LS_dict = fit_LS_model_and_find_priors(s_data, design, info_dict, mean_only)

    # find parametric adjustments
    if eb:
        if parametric:
            print('[neuroCombat] Finding parametric adjustments')
            gamma_star, delta_star = find_parametric_adjustments(s_data, LS_dict, info_dict, mean_only)
        else:
            print('[neuroCombat] Finding non-parametric adjustments')
            gamma_star, delta_star = find_non_parametric_adjustments(s_data, LS_dict, info_dict, mean_only)
    else:
        print('[neuroCombat] Finding L/S adjustments without Empirical Bayes')
        gamma_star, delta_star = find_non_eb_adjustments(s_data, LS_dict, info_dict)

    # adjust data
    print('[neuroCombat] Final adjustment of data')
    bayes_data = adjust_data_final(s_data, design, gamma_star, delta_star, 
                                    s_mean, mod_mean, v_pool, info_dict,dat)

    bayes_data = np.array(bayes_data)
    estimates = {'batches': info_dict['batch_levels'], 'var.pooled': v_pool, 'stand.mean': s_mean, 'mod.mean': mod_mean, 'gamma.star': gamma_star, 'delta.star': delta_star}
    estimates = {**LS_dict, **estimates, }

    return {
        'data': bayes_data,
        'estimates': estimates,
        'info': info_dict
    }




def make_design_matrix(Y, batch_col, cat_cols, num_cols, ref_level):
    """
    Return Matrix containing the following parts:
        - one-hot matrix of batch variable (full)
        - one-hot matrix for each categorical_cols (removing the first column)
        - column for each continuous_cols
    """
    def to_categorical(y, nb_classes=None):
        if not nb_classes:
            nb_classes = np.max(y)+1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y
    
    hstack_list = []

    ### batch one-hot ###
    # convert batch column to integer in case it's string
    batch = np.unique(Y[:,batch_col],return_inverse=True)[-1]
    batch_onehot = to_categorical(batch, len(np.unique(batch)))
    if ref_level is not None:
        batch_onehot[:,ref_level] = np.ones(batch_onehot.shape[0])
    hstack_list.append(batch_onehot)

    ### categorical one-hots ###
    for cat_col in cat_cols:
        cat = np.unique(np.array(Y[:,cat_col]),return_inverse=True)[1]
        cat_onehot = to_categorical(cat, len(np.unique(cat)))[:,1:]
        hstack_list.append(cat_onehot)

    ### numerical vectors ###
    for num_col in num_cols:
        num = np.array(Y[:,num_col],dtype='float32')
        num = num.reshape(num.shape[0],1)
        hstack_list.append(num)

    design = np.hstack(hstack_list)
    return design


def standardize_across_features(X, design, info_dict):
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    sample_per_batch = info_dict['sample_per_batch']
    batch_info = info_dict['batch_info']
    ref_level = info_dict['ref_level']

    def get_beta_with_nan(yy, mod):
        wh = np.isfinite(yy)
        mod = mod[wh,:]
        yy = yy[wh]
        B = np.dot(np.dot(la.inv(np.dot(mod.T, mod)), mod.T), yy.T)
        return B

    betas = []
    for i in range(X.shape[0]):
        betas.append(get_beta_with_nan(X[i,:], design))
    B_hat = np.vstack(betas).T
    
    #B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), X.T)
    if ref_level is not None:
        grand_mean = np.transpose(B_hat[ref_level,:])
    else:
        grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    #var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))

    if ref_level is not None:
        X_ref = X[:,batch_info[ref_level]]
        design_ref = design[batch_info[ref_level],:]
        n_sample_ref = sample_per_batch[ref_level]
        var_pooled = np.dot(((X_ref - np.dot(design_ref, B_hat).T)**2), np.ones((n_sample_ref, 1)) / float(n_sample_ref))
    else:
        var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))

    var_pooled[var_pooled==0] = np.median(var_pooled!=0)
    
    mod_mean = 0
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(0,n_batch)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))
    ######### Continue here. 


    #tmp = np.array(design.copy())
    #tmp[:,:n_batch] = 0
    #stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((X- stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled, mod_mean

def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def convert_zeroes(x):
    x[x==0] = 1
    return x

def fit_LS_model_and_find_priors(s_data, design, info_dict, mean_only):
    n_batch = info_dict['n_batch']
    batch_info = info_dict['batch_info'] 
    
    batch_design = design[:,:n_batch]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []
    for i, batch_idxs in enumerate(batch_info):
        if mean_only:
            delta_hat.append(np.repeat(1, s_data.shape[0]))
        else:
            delta_hat.append(np.var(s_data[:,batch_idxs],axis=1,ddof=1))
    
    delta_hat = list(map(convert_zeroes,delta_hat))
    gamma_bar = np.mean(gamma_hat, axis=1) 
    t2 = np.var(gamma_hat,axis=1, ddof=1)

    if mean_only:
        a_prior = None
        b_prior = None
    else:
        a_prior = list(map(aprior, delta_hat))
        b_prior = list(map(bprior, delta_hat))

    LS_dict = {}
    LS_dict['gamma_hat'] = gamma_hat
    LS_dict['delta_hat'] = delta_hat
    LS_dict['gamma_bar'] = gamma_bar
    LS_dict['t2'] = t2
    LS_dict['a_prior'] = a_prior
    LS_dict['b_prior'] = b_prior
    return LS_dict

#Helper function for parametric adjustements:
def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)

        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 



#Helper function for non-parametric adjustements:
def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0,r,1):
        g = np.delete(g_hat,i)
        d = np.delete(d_hat,i)
        x = sdat[i,:]
        n = x.shape[0]
        j = np.repeat(1,n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n,g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0],n)
        resid2 = np.square(A-B)
        sum2 = resid2.dot(j)
        LH = 1/(2*math.pi*d)**(n/2)*np.exp(-sum2/(2*d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g*LH)/sum(LH))
        delta_star.append(sum(d*LH)/sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust


def find_parametric_adjustments(s_data, LS, info_dict, mean_only):
    batch_info  = info_dict['batch_info'] 
    ref_level = info_dict['ref_level']

    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        if mean_only:
            gamma_star.append(postmean(LS['gamma_hat'][i], LS['gamma_bar'][i], 1, 1, LS['t2'][i]))
            delta_star.append(np.repeat(1, s_data.shape[0]))
        else:
            temp = it_sol(s_data[:,batch_idxs], LS['gamma_hat'][i],
                        LS['delta_hat'][i], LS['gamma_bar'][i], LS['t2'][i], 
                        LS['a_prior'][i], LS['b_prior'][i])
            gamma_star.append(temp[0])
            delta_star.append(temp[1])

    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    if ref_level is not None:
        gamma_star[ref_level,:] = np.zeros(gamma_star.shape[-1]) 
        delta_star[ref_level,:] = np.ones(delta_star.shape[-1]) 

    return gamma_star, delta_star

def find_non_parametric_adjustments(s_data, LS, info_dict, mean_only):
    batch_info  = info_dict['batch_info'] 
    ref_level = info_dict['ref_level']

    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        if mean_only:
            LS['delta_hat'][i] = np.repeat(1, s_data.shape[0])
        temp = int_eprior(s_data[:,batch_idxs], LS['gamma_hat'][i],
                    LS['delta_hat'][i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    if ref_level is not None:
        gamma_star[ref_level,:] = np.zeros(gamma_star.shape[-1]) 
        delta_star[ref_level,:] = np.ones(delta_star.shape[-1]) 

    return gamma_star, delta_star

def find_non_eb_adjustments(s_data, LS, info_dict):
    gamma_star = np.array(LS['gamma_hat'])
    delta_star = np.array(LS['delta_hat'])
    ref_level = info_dict['ref_level']
    
    if ref_level is not None:
        gamma_star[ref_level,:] = np.zeros(gamma_star.shape[-1]) 
        delta_star[ref_level,:] = np.ones(delta_star.shape[-1])
    
    return gamma_star, delta_star

def adjust_data_final(s_data, design, gamma_star, delta_star, stand_mean, mod_mean, var_pooled, info_dict, dat):
    sample_per_batch = info_dict['sample_per_batch']
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    batch_info = info_dict['batch_info']
    ref_level = info_dict['ref_level']

    batch_design = design[:,:n_batch]

    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    for j, batch_idxs in enumerate(batch_info):
        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom = np.dot(dsq, np.ones((1, sample_per_batch[j])))
        numer = np.array(bayesdata[:,batch_idxs] - np.dot(batch_design[batch_idxs,:], gamma_star).T)

        bayesdata[:,batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, n_sample))) + stand_mean + mod_mean

    if ref_level is not None:
        bayesdata[:, batch_info[ref_level]] = dat[:,batch_info[ref_level]]

    return bayesdata




def neuroCombatFromTraining(dat,
                            batch,
                            estimates):
    """
    Combat harmonization with pre-trained ComBat estimates [UNDER DEVELOPMENT]

    Arguments
    ---------
    dat : a pandas data frame or numpy array for the new dataset to harmonize
        - rows must be identical to the training dataset
    
    batch : numpy array specifying scanner/batch for the new dataset
        - scanners/batches must also be present in the training dataset

    estimates : dictionary of ComBat estimates from a previously-harmonized dataset
        - should be in the same format as neuroCombat(...)['estimates']
        
    Returns
    -------
    A dictionary of length 2:
    - data: A numpy array with the same shape as `dat` which has now been ComBat-harmonized
    - estimates: A dictionary of the ComBat estimates used for harmonization
    """
    print("[neuroCombatFromTraining] In development ...\n")
    batch = np.array(batch, dtype="str")
    new_levels = np.unique(batch)
    old_levels = np.array(estimates['batches'], dtype="str")
    missing_levels = np.setdiff1d(new_levels, old_levels)
    if missing_levels.shape[0] != 0:
        raise ValueError("The batches " + str(missing_levels) +
                         " are not part of the training dataset")


    wh = [int(np.where(old_levels==x)[0]) if x in old_levels else None for x in batch]

    

    var_pooled = estimates['var.pooled']
    stand_mean = estimates['stand.mean'][:, 0]
    mod_mean = estimates['mod.mean']
    gamma_star = estimates['gamma.star']
    delta_star = estimates['delta.star']
    n_array = dat.shape[1]   
    stand_mean = stand_mean+mod_mean.mean(axis=1)
 
    
    stand_mean = np.transpose([stand_mean, ]*n_array)
    bayesdata = np.subtract(dat, stand_mean)/np.sqrt(var_pooled)
    
    #gamma = np.transpose(np.repeat(gamma_star, repeats=2, axis=0))
    #delta = np.transpose(np.repeat(delta_star, repeats=2, axis=0))
    gamma = np.transpose(gamma_star[wh,:])
    delta = np.transpose(delta_star[wh,:])
    bayesdata = np.subtract(bayesdata, gamma)/np.sqrt(delta)
    
    bayesdata = bayesdata*np.sqrt(var_pooled) + stand_mean
    out = {
        'data': bayesdata,
        'estimates': estimates
    }
    return out
