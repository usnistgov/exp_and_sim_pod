import pandas as pd
import numpy as np
import pystan
import arviz as az
from sklearn import linear_model

exp_data = pd.read_excel('../data/a^_vs_a_image_threshold_v2.xlsx')
exp_data['sim'] = np.repeat(np.nan, exp_data.shape[0])
exp_data['source'] = np.repeat('exp', exp_data.shape[0])

sim_data = pd.read_csv('../data/simulation0.csv', header=None)
for i in range(11):
    sim_data = pd.concat([sim_data,
                          pd.read_csv('../data/simulation' +
                                      str(i+1) + '.csv',
                                      header=None)])

sim_data['a'] = sim_data.iloc[:, 2]*1000*40/(1000**3)
sim_data['a_unc'] = np.zeros((sim_data.shape[0],))
sim_data['sim'] = np.repeat(np.linspace(1, 12, 12), 20)
sim_data['source'] = np.repeat('sim', 12*20)

exp_data = (exp_data[['a (mm3)', 'a_unc (mm3)', 'sim', 'source',
                      'th1 voxel', 'th2 voxel', 'th3 voxel']].
            rename(columns={'a (mm3)': 'a',
                            'a_unc (mm3)': 'a_unc',
                            'th1 voxel': 1,
                            'th2 voxel': 2,
                            'th3 voxel': 3}))

sim_data = (sim_data[['a', 'a_unc', 'sim', 'source', 5, 6, 7]].
            rename(columns={5: 1,
                            6: 2,
                            7: 3}))

full_data = pd.concat([exp_data, sim_data])

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

stan_model = pystan.StanModel('./model_with_sim_neg_binom_quad.stan')

N_exp = np.sum(full_data['source'] == 'exp')

stan_data = {'N_exp': int(N_exp),
             'N_non_exp': int(full_data.shape[0] - N_exp),
             'N_thresh_group': int(np.max(full_data['thresh_rep'])),
             'N_sim': int(np.max(full_data['sim'])),
             'voxels_exp': full_data['Voxels'][full_data['source'] == 'exp'].astype(np.int),
             'voxels_non_exp': full_data['Voxels'][full_data['source'] == 'sim'].astype(np.int),
             'a_exp': full_data['a'][full_data['source'] == 'exp'],
             'a_non_exp': full_data['a'][full_data['source'] == 'sim'],
             'a_exp_unc': full_data['a_unc'][full_data['source'] == 'exp'],
             'thresh_exp': full_data['thresh_cen'][full_data['source'] == 'exp'],
             'thresh_non_exp': full_data['thresh_cen'][full_data['source'] == 'sim'],
             'thresh_group_exp': full_data['thresh_rep'][full_data['source'] == 'exp'].astype(np.int),
             'thresh_group_non_exp': full_data['thresh_rep'][full_data['source'] == 'sim'].astype(np.int),
             'sim_id': full_data['sim'][full_data['source'] == 'sim'].astype(np.int),
             'N_pod': 200,
             'a_pod': np.exp(np.linspace(np.log(0.00001), np.log(0.005), 200)),
             'N_voxel_threshold': 4,
             'voxel_threshold': pd.Series([50, 100, 200, 300]),
             'N_pred': 200,
             'a_pred': np.exp(np.linspace(np.log(0.0001), np.log(0.15), 200))}

y = np.log(full_data['Voxels'])
X = np.array([np.log(full_data['a']),
              np.log(full_data['a'])*np.log(full_data['a'])]).T
lm_fit = linear_model.LinearRegression().fit(X, y)
stan_init = [{'a_int_mean': float(lm_fit.intercept_),
              'a_slope_mean': float(lm_fit.coef_[0]),
              'a_quad_mean': float(lm_fit.coef_[1]),
              'a_int_sd': float(np.abs(lm_fit.intercept_)*0.1),
              'a_slope_sd': float(np.abs(lm_fit.coef_[0])*0.1),
              'a_quad_sd': float(np.abs(lm_fit.coef_[1])*0.1),
              'thesh_slope_mean': 0.4,
              'shresh_slope_sd': 0.1,
              'la_exp_true': np.log(stan_data['a_exp']),
              'a_int': np.repeat(float(lm_fit.intercept_), stan_data['N_sim'] + 1),
              'a_slope': np.repeat(float(lm_fit.coef_[0]), stan_data['N_sim'] + 1),
              'a_quad': np.repeat(float(lm_fit.coef_[1]), stan_data['N_sim'] + 1),
              'thresh_slope': np.repeat(0.4, stan_data['N_thresh_group'])}]*5

post_samps = stan_model.sampling(data=stan_data,
                                 iter=2000,
                                 warmup=1000,
                                 chains=5,
                                 control={'max_treedepth': 17,
                                          'adapt_delta': 0.999999},
                                 init=stan_init,
                                 n_jobs=2)

post_samps = az.from_pystan(posterior=post_samps)

az.to_netcdf(post_samps, 'post_samps_with_sim_neg_binom_quad.nc')

