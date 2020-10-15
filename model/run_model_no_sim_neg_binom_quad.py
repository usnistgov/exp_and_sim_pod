import pandas as pd
import numpy as np
import pystan
import matplotlib.pyplot as plt
import arviz as az
from sklearn import linear_model

exp_data = pd.read_excel('../data/a^_vs_a_image_threshold_v2.xlsx')
exp_data['sim'] = np.repeat(np.nan, exp_data.shape[0])
exp_data['source'] = np.repeat('exp', exp_data.shape[0])

exp_data = (exp_data[['a (mm3)', 'a_unc (mm3)', 'sim', 'source',
                      'th1 voxel', 'th2 voxel', 'th3 voxel']].
            rename(columns={'a (mm3)': 'a',
                            'a_unc (mm3)': 'a_unc',
                            'th1 voxel': 1,
                            'th2 voxel': 2,
                            'th3 voxel': 3}))

full_data = exp_data

full_data = full_data.set_index(['a', 'a_unc', 'sim', 'source'])

full_data = full_data.stack(dropna=False)

full_data = full_data.reset_index()

full_data = full_data.rename(columns={'level_4': 'thresh',
                                      0: 'Voxels'})

full_data['thresh'] = full_data['thresh'].astype(np.float)
full_data['thresh_cen'] = full_data['thresh'].copy() - 2.0

full_data['thresh_rep'] = (
    np.repeat(
        np.linspace(1, int(full_data.shape[0]/3),
                    int(full_data.shape[0]/3)), 3
    )
)
full_data['thresh_rep'] = full_data['thresh_rep'].astype(np.int)

full_data = full_data[~np.isnan(full_data['Voxels'])]

stan_model = pystan.StanModel('./model_no_sim_neg_binom_quad.stan')

N_exp = np.sum(full_data['source'] == 'exp')

stan_data = {'N_exp': int(N_exp),
             'N_thresh_group': int(np.max(full_data['thresh_rep'])),
             'voxels_exp': full_data['Voxels'][full_data['source'] == 'exp'].astype(np.int),
             'a_exp': full_data['a'][full_data['source'] == 'exp'],
             'a_exp_unc': full_data['a_unc'][full_data['source'] == 'exp'],
             'thresh_exp': full_data['thresh_cen'][full_data['source'] == 'exp'],
             'thresh_group_exp': full_data['thresh_rep'][full_data['source'] == 'exp'].astype(np.int),
             'N_pod': 200,
             'a_pod': np.exp(np.linspace(np.log(0.00001), np.log(0.005), 200)),
             'N_voxel_threshold': 4,
             'voxel_threshold': pd.Series([50, 100, 200, 300]).astype(np.int),
             'N_pred': 200,
             'a_pred': np.exp(np.linspace(np.log(0.0001), np.log(0.15), 200))}

y = np.log(full_data['Voxels'])
X = np.array([np.log(full_data['a']),
              np.log(full_data['a'])*np.log(full_data['a'])]).T
lm_fit = linear_model.LinearRegression().fit(X, y)
stan_init = [{'a_int': float(lm_fit.intercept_),
              'a_slope': float(lm_fit.coef_[0]),
              'a_quad': float(lm_fit.coef_[1]),
              'thesh_slope_mean': -0.1,
              'shresh_slope_sd': 0.05,
              'la_exp_true': np.log(stan_data['a_exp']),
              'thresh_slope': np.repeat(0.4, stan_data['N_thresh_group'])}]*5

post_samps = stan_model.sampling(data=stan_data,
                                 iter=2000,
                                 chains=5,
                                 control={'max_treedepth': 15, 'adapt_delta': 0.999999},
                                 init=stan_init)

post_samps = az.from_pystan(posterior=post_samps,
                            coords={'thresh_group': np.arange(stan_data['N_thresh_group']),
                                    'obs': np.arange(full_data.shape[0]),
                                    'N_pred': np.arange(200),
                                    'N_pod': np.arange(200)},
                            dims={'thresh_slope': ['thresh_group'],
                                  'la_exp_true': ['obs'],
                                  'voxels_pred': ['N_pred'],
                                  'prob_thresh': ['N_pod']})

az.to_netcdf(post_samps, 'post_samps_no_sim_neg_binom_quad.nc')

