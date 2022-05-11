from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

from .neuroCombat import (aprior, bprior, int_eprior,
                          it_sol, convert_zeroes)

class Combat(BaseEstimator, TransformerMixin):

    _needs_fit_index = True
    _needs_transform_index = True
    
    def __init__(self, batch, covars=None,
                 include_y=False, y_is_cat=False,
                 eb=True, parametric=True, use_mean_for_new=False,
                 verbose=False):
        ''' Sklearn-style class for performing ComBat geared towards removing scanner
        effects in multi-site imaging data.

        Parameters
        ----------
        batch : pandas Series or str
            A pandas series containing the batch information to use, e.g., "scanner".
            These values can be strings or already ordinally encoded. Can also optionally
            pass batch as a str, where in this case it would refer to the name of
            a column in the passed covars.

        covars : pandas DataFrame or None, optional
            A DataFrame containing any optional covariates
            that should be preserved during harmonization. If batch is
            passed as a str, then this DataFrame should also contain a column containing
            the column w/ batch/scanner info.

            Note: Any covariates which are categorical, that is to say, those that
            still require dummy coding, should have 'category' as their dtype, this
            can be set in pandas w/ df['col'] = df['col'].astype('category').
            default = None.

        include_y : bool, optional
            This parameter specifies if y, the target variable when fitting, should
            be treated as one of the covariates to try and preserve. Note that is this
            is set to True, you also must set y_is_cat, and keep in mind that when predicting
            on new samples, e.g., transform is called on new samples, the mean of y in the training
            sample is used (as using y directly on the test set would be a breach of cross-validation).
            default = False.

        y_is_cat : bool, optional
            This parameter let's Combat know if the target variable y is categorical or not.
            If True, then it will dummy-coded internally when used for preserving its value,
            otherwise if False, the value as is will be used.
            default = False

        eb : bool, optional
            This parameter controls if Empirical Bayes be performed.
            default = True

        parametric : bool, optional
            This parameter controls if parameter adjustments should be performed.
            default = True

        use_mean_for_new : bool, optional
            This parameter describes the behavior of Combat when predicting
            on new samples with respect to how covariates are handled. Specifically,
            if True, then behavior mimics the function `neuroCombatFromTraining`,
            and covariate corrections on new samples just use the mean value from those
            covariates in the training set. Alternatively, the default behavior is to
            instead base the correction off the actual covariate values in the new sample.

            Note that this parameter refers to all other covariates except y, as y is a special
            case, where in the context of new samples we have to use the mean estimate.

        verbose : bool, optional
            If True, then print verbose messages.
            default = False.
        
        '''
        
        self.batch = batch
        self.covars = covars
        self.include_y  = include_y
        self.y_is_cat = y_is_cat
        self.eb = eb
        self.parametric = parametric
        self.use_mean_for_new = use_mean_for_new
        self.verbose = verbose

        self._check_args()

    def _check_args(self):

        if not isinstance(self.covars, pd.DataFrame) and self.covars is not None:
            raise ValueError('covars must be pandas dataframe or None!')
        
        # If batch series is str, make sure exists in covars
        if isinstance(self.batch, str):
            if self.covars is None:
                raise ValueError('If batch is a str, then you must pass a DataFrame for covars.')

            if self.batch not in self.covars:
                raise ValueError(f'batch={self.batch} not found in covars, either pass as Series or valid column: {list(self.covars)}')

        if not isinstance(self.batch, (pd.Series, str)):
            raise ValueError('batch must be a pandas Series or str')
    
    @property
    def batch_series(self):

        # If column in covars df
        if isinstance(self.batch, str):
            return self.covars[self.batch]

        # If already series
        return self.batch

    @property
    def covars_df(self):

        # If batch is a column in covars
        if isinstance(self.batch, str):

            # Get covars df w/o batch column
            covars_df = self.covars.drop(self.batch, axis=1)

            # Return None if now empty
            if covars_df.shape[1] == 0:
                return None
            return covars_df
        
        # Otherwise, return as is
        return self.covars

    def _add_y_series(self, y, fit, covars, index):

        # If y is not None, then either in fit / fit_transform case.
        if y is not None:
            
            # Set y as a series
            y_series = pd.Series(y)

            # Then if, fitting, want to save info on y_mean
            if fit:
                
                # Regression case is easier, just keep y_mean
                if not self.y_is_cat:
                    self.y_mean_ = np.mean(y)

                # Otherwise, if y is categorical, then we want to dummy code y
                else:

                    self.y_encoder_ = OneHotEncoder(drop='first', sparse=False)
                    y_encoded = self.y_encoder_.fit_transform(np.array(y.to_frame()))
                    self.y_mean_  = y_encoded.mean(axis=0)

            # The non fit case is used in fit_transform too
            else:

                # Encode y based on already computed
                if self.y_is_cat:
                    y_encoded = self.y_encoder_.transform(np.array(y.to_frame()))

        # If y isn't passed, then in transform new case
        else:
            if self.y_is_cat:
                y_encoded = np.tile(self.y_mean_, len(index)).reshape((len(index), -1))
            else:
                y_series = pd.Series([self.y_mean_ for _ in range(len(index))])

        # Now add either y_series to df or y_encoded as multiple columns
        # y_encoded will be init'ed if y is cat
        if self.y_is_cat:
            covars[[f'y_{i}' for i in range(y_encoded.shape[1])]] = y_encoded
        else:
            covars['y'] = y_series

        return covars
    
    def _transform_covars(self, index, y=None, fit=True):

        # If None, and no include y, return None
        if not self.include_y and self.covars_df is None:
            return None

        # Init covars as either empty or based  current covars at this index
        if self.covars_df is not None:
            covars = self.covars_df.loc[index].copy()
        else:
            covars = pd.DataFrame()

        # If include y, and y is not None, i.e., in fit case, not transform
        if self.include_y:
            covars = self._add_y_series(y, fit, covars, index)

        # Find which cols are categorical based on dtype
        cat_cols = [col for col in covars
                    if covars[col].dtype.name == 'category']

        # If any categorical, dummy-code
        if len(cat_cols) > 0:

            # Cast to np
            cat_trans = np.array(covars[cat_cols])

            # Init and fit, if fit
            if fit:
                self.encoder_ = OneHotEncoder(drop='first', sparse=False)
                self.encoder_.fit(cat_trans)

            # Transform
            cat_trans = self.encoder_.transform(cat_trans)

            # Stack with non-cat in place
            rest_cols = [col for col in covars if col not in cat_cols]
            covars = np.hstack([cat_trans, np.array(covars[rest_cols])])

        # At this stage make sure np array if not already
        covars = np.array(covars)

        return covars

    def _get_batch(self, index, fit=True):

        # One-hot encode batch_series
        batch_col = np.array(self.batch_series.loc[index])
        batch_col = np.reshape(batch_col, (-1, 1))
        
        # Fit encoder if needed
        if fit:
            self.batch_encoder_ = OneHotEncoder(drop=None, sparse=False)
            self.batch_encoder_.fit(batch_col)
            
        # Then one-hot encode
        batch_one_hot = self.batch_encoder_.transform(batch_col)

        return batch_one_hot
    
    def _get_design_matrix(self, index, y=None, fit=True):
        '''Get design matrix as one-hot of batch series + dummy coded categorical
        covars + cont.'''
        
        # One-hot encode batch_series
        batch_one_hot = self._get_batch(index, fit=fit)
        
        # Get transformed covars if any - if in fit and include y is passed, then
        # will also get y
        covars_trans = self._transform_covars(index=index,  y=y, fit=fit)
        
        # Return
        if covars_trans is not None:
            return np.hstack([batch_one_hot, covars_trans])
        return batch_one_hot
    
    def _calc_mod_mean(self, design):

        if self.covars_df is None and self.include_y is False:
            return 0

        return np.dot(design[:, self.n_batch_:], self.b_hat_[self.n_batch_:,]).T

    def _standardize_across_feats(self, Xt, design):

        # Get beta weights
        self.b_hat_ = get_b_hat(Xt, design)
        
        # Calculate grand mean, stand mean and var pooled
        self.grand_mean_ = np.dot((self.sample_per_batch_ / float(self.n_sample_)).T,
                                   self.b_hat_[: self.n_batch_, :])

        # Just  re-shaped grand mean
        stand_mean = np.repeat(self.grand_mean_, self.n_sample_).reshape(-1, self.n_sample_)
        
        self.var_pooled_ = np.dot(((Xt - np.dot(design, self.b_hat_).T)**2),
                            np.ones((self.n_sample_, 1)) / float(self.n_sample_))
        self.var_pooled_[self.var_pooled_ == 0] = np.median(self.var_pooled_ != 0)

        # Calc mod mean
        self.mod_mean_ = self._calc_mod_mean(design)
        
        # Calc stand. data
        s_data = (Xt - stand_mean - self.mod_mean_) / np.sqrt(self.var_pooled_)
        
        return s_data

    def _fit_ls_and_find_priors(self, s_data, design):

        batch_design = design[:, :self.n_batch_]
        self.gamma_hat_ = np.dot(np.dot(np.linalg.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

        self.delta_hat_ = []
        for _, batch_idxs in enumerate(self.batch_info_):
            self.delta_hat_.append(np.var(s_data[:, batch_idxs], axis=1, ddof=1))
        
        self.delta_hat_ = list(map(convert_zeroes, self.delta_hat_))
        self.gamma_bar_ = np.mean(self.gamma_hat_, axis=1) 
        self.t2_ = np.var(self.gamma_hat_,axis=1, ddof=1)

        self.a_prior_ = list(map(aprior, self.delta_hat_))
        self.b_prior_ = list(map(bprior, self.delta_hat_))

    def _find_parametric_adjustments(self, s_data):

        gamma_star, delta_star = [], []
        for i, batch_idxs in enumerate(self.batch_info_):

            temp = it_sol(s_data[: ,batch_idxs], self.gamma_hat_[i], self.delta_hat_[i],
                          self.gamma_bar_[i], self.t2_[i], self.a_prior_[i], self.b_prior_[i])
            gamma_star.append(temp[0])
            delta_star.append(temp[1])

        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)

        return gamma_star, delta_star

    def _find_non_parametric_adjustments(self, s_data):
    
        gamma_star, delta_star = [], []
        for i, batch_idxs in enumerate(self.batch_info_):
            temp = int_eprior(s_data[:,batch_idxs], self.gamma_hat_[i], self.delta_hat_[i])
            gamma_star.append(temp[0])
            delta_star.append(temp[1])

        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)

        return gamma_star, delta_star

    def _set_adjustments(self, s_data):

        # Emperical Bayes adjustment w/ or w/o parameter
        if self.eb:
            if self.parametric:
                if self.verbose:
                    print('Finding parametric adjustments')
                self.gamma_star_, self.delta_star_ = self._find_parametric_adjustments(s_data)
            else:
                if self.verbose:
                    print('Finding non-parametric adjustments')
                self.gamma_star_, self.delta_star_ = self._find_non_parametric_adjustments(s_data)
        
        # Non EB adjustment case, use estimates directly
        else:
            if self.verbose:
                print('Finding L/S adjustments without Empirical Bayes')
            self.gamma_star_, self.delta_star_ = np.array(self.gamma_hat_), np.array(self.delta_hat_)
            
    def fit(self, X, y=None, fit_index=None):
        
        # Input checks
        self._check_args()
        
        if isinstance(X, pd.DataFrame):
            if fit_index is None:
                fit_index = X.index
            X = np.array(X)
                
        # Set attributes
        self.batch_levels_, self.sample_per_batch_ =\
            np.unique(self.batch_series.loc[fit_index], return_counts=True)
        
        self.n_batch_ = len(self.batch_levels_)
        self.n_sample_ = len(X)
        
        batch_ref = np.array(self.batch_series.loc[fit_index])
        self.batch_info_ = [list(np.where(batch_ref == idx)[0])
                            for idx in self.batch_levels_]
        
        # Get batch + covars together as design matrix
        design = self._get_design_matrix(index=fit_index,  y=y, fit=True)
        if self.verbose:
            print(f'Created design matrix w/ shape: {design.shape}')
    
        # Standardize
        if self.verbose:
            print('Standardize across features')
        s_data = self._standardize_across_feats(X.T.copy(), design)

        # Fit ls and find priors
        if self.verbose:
            print('Fitting L/S model and finding priors.')
        self._fit_ls_and_find_priors(s_data, design)

        # Find adjustments, sets gamma and delta star
        self._set_adjustments(s_data)
        
        return self

    def _transform(self, X, transform_index=None, is_fit=False):

        # Get design matrix for transfrom data, fit=False and y fixed as None
        # Note that is include_y is True
        design = self._get_design_matrix(transform_index, y=None, fit=False)

        # Standerdize data - changes if this is on the same fit data
        # If this is on the fit data than we can use the already calculated mod mean
        if is_fit:
            dif = np.repeat(self.grand_mean_, len(X)).reshape(-1, len(X)) + self.mod_mean_

        # Otherwise, we can compute new mod mean based on the data to predicts covar values
        else:

            # Either use stored mod mean from fit
            if self.use_mean_for_new:
                dif = self.grand_mean_ + self.mod_mean_.mean(axis=1)
                dif = np.transpose([dif, ] * len(X))

            # Or calculate it new
            else:
                dif = np.repeat(self.grand_mean_, len(X)).reshape(-1, len(X)) +\
                    self._calc_mod_mean(design)

        # Now we get the actual standardized data here
        # Note, for fit this is duplicated computation, but not a big deal
        X_trans = (X.T - dif) / np.sqrt(self.var_pooled_)

        # Get reference batch cols
        batch_cols = design[:, :self.n_batch_]
        wh = np.where(batch_cols == 1)[1]

        # Apply the estimated gamma and delta (gamma / delta vals as computed per batch)
        gamma = self.gamma_star_[wh, :].T
        delta = self.delta_star_[wh, :].T
        X_trans = np.subtract(X_trans, gamma) / np.sqrt(delta)

        # Lastly, we need to reverse apply the pooled variance / dif
        X_trans = X_trans * np.sqrt(self.var_pooled_) + dif

        # Make sure to transpose the results
        return X_trans.T        
    
    def transform(self, X, transform_index=None):

        # Convert from pandas df if needed
        if isinstance(X, pd.DataFrame):
            if transform_index is None:
                transform_index = X.index
            X = np.array(X)

        return self._transform(X, transform_index=transform_index, is_fit=False)
    
    def fit_transform(self, X, y=None, fit_index=None):

        if isinstance(X, pd.DataFrame):
            if fit_index is None:
                fit_index = X.index
            X = np.array(X)

        # Pass same fit index to both, and make sure is_fit is passed to transform
        return self.fit(X=X, y=y,
                        fit_index=fit_index)._transform(X=X, is_fit=True,
                                                        transform_index=fit_index)

    def __repr__(self):

        # TODO change rep
        return 'Combat(...)'

def get_beta_with_nan(yy, mod):
    
    wh = np.isfinite(yy)
    mod = mod[wh, :]
    yy = yy[wh]

    # If singular matrix error, try getting the pseudo inverse instead
    # and use that.
    try:
        inv = np.linalg.inv(np.dot(mod.T, mod))
    except LinAlgError:
        inv = np.linalg.pinv(np.dot(mod.T, mod))

    B = np.dot(np.dot(inv, mod.T), yy.T)

    return B

def get_b_hat(X, design):
    
    betas = []
    for i in range(X.shape[0]):
        betas.append(get_beta_with_nan(X[i, :], design))
    
    return np.vstack(betas).T



