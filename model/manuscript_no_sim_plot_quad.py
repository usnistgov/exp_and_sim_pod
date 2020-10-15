import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

post_samps = az.from_netcdf('./post_samps_no_sim_neg_binom_quad.nc')

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

full_data['thresh'] = full_data['thresh'].astype('category')
full_data['sim'] = full_data['sim'].astype('category')

full_data['thresh_rep'] = (
    np.repeat(
        np.linspace(1, int(full_data.shape[0]/3),
                    int(full_data.shape[0]/3)), 3
    )
)
full_data['thresh_rep'] = full_data['thresh_rep'].astype('category')

full_data = full_data[~np.isnan(full_data['Voxels'])]

grand_int = np.median(post_samps.posterior.a_int)
grand_slope = np.median(post_samps.posterior.a_slope)
full_data['lambda_est'] = np.exp(grand_int +
                                 grand_slope*np.log(full_data['a']))

est_bounds = pd.DataFrame()
est_bounds['lb'] = np.quantile(post_samps.posterior.voxels_pred, q=0.025, axis=(0, 1))
est_bounds['ub'] = np.quantile(post_samps.posterior.voxels_pred, q=0.975, axis=(0, 1))
est_bounds['Voxels'] = np.quantile(post_samps.posterior.voxels_pred, q=0.5, axis=(0, 1))
est_bounds['a'] = np.exp(np.linspace(np.log(0.0001), np.log(0.15), 200))
est_bounds['thresh'] = 1
est_bounds['thresh'] = est_bounds['thresh'].astype('category')
est_bounds['source'] = 'exp'
est_bounds['lb_diff'] = est_bounds['Voxels'] - est_bounds['lb']
est_bounds['ub_diff'] = est_bounds['ub'] - est_bounds['Voxels']

pod_data = pd.DataFrame()
pod_data['est'] = np.quantile(post_samps.posterior.prob_thresh[:, :, :, 1],
                              q=0.5, axis=(0, 1))
pod_data['lb'] = np.quantile(post_samps.posterior.prob_thresh[:, :, :, 1],
                             q=0.025, axis=(0, 1))
pod_data['ub'] = np.quantile(post_samps.posterior.prob_thresh[:, :, :, 1],
                             q=0.975, axis=(0, 1))
pod_data['a'] = np.exp(np.linspace(np.log(0.00001), np.log(0.005), 200))
pod_data['thresh'] = 1
pod_data['thresh'] = est_bounds['thresh'].astype('category')
pod_data['source'] = 'exp'

plt_char=['o', '^', 's']
thresh_labels = ['Min. Threshold', 'Mid.', 'Max.']
fig, ax = plt.subplots(ncols=2)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].fill_between(est_bounds.a, est_bounds['lb'], est_bounds['ub'],
                   color='lightgrey')
ax[0].plot(est_bounds.a, est_bounds.Voxels, 'k-')
for i in range(3):
    ax[0].scatter(x=full_data.a[full_data.thresh==(i+1)],
                  y=full_data.Voxels[full_data.thresh==(i+1)],
                  marker=plt_char[i], c='black',
                  s=7500*full_data.a_unc[full_data.thresh==(i+1)],
                  label=thresh_labels[i])
ax[0].legend(frameon=False)
ax[0].set_xlabel('$\\tilde{a}$ (mm$^3$)')
ax[0].set_ylabel('Voxel Count')
ax[0].annotate('(a)', (0.85, 0.15), xycoords='axes fraction')
ax[0].annotate('Size proportional to', (0.2, 0.075), xycoords='axes fraction')
ax[0].annotate('uncertainty in $\\tilde{a}$', (0.2, 0.025), xycoords='axes fraction')
ax[1].plot(pod_data.a, pod_data.est, 'k-')
ax[1].fill_between(pod_data.a, pod_data.lb, pod_data.ub,
                   color='lightgrey')
ax[1].set_xscale('log')
ax[1].set_xlabel('$a$ (mm$^3$)')
ax[1].set_ylabel('POD')
ax[1].annotate('(b)', (0.85, 0.15), xycoords='axes fraction')
plt.tight_layout()

plt.savefig('no_sim_results_manuscript_neg_binom_quad.png', dpi=600)

tmp = pd.merge_asof(full_data.sort_values('a'), est_bounds.sort_values('a'), on='a')
tmp = tmp[['a', 'Voxels_x', 'lb', 'ub']]
tmp['in_interval'] = (tmp['Voxels_x'] > tmp['lb']) & (tmp['Voxels_x'] < tmp['ub'])
print(tmp.mean())
print(tmp.iloc[6:, :].mean())

# traditional POD curve with parametric bootstrap for uncertainty

def calc_pod(a, Vc, beta0, beta1, beta2, sigma):
    mu = beta0 + beta1*np.log(a) + beta2*np.log(a)*np.log(a)
    return 1 - norm.cdf(np.log(Vc), mu, sigma)

log_a = np.log(full_data['a'])
lm_X = np.array([log_a, log_a*log_a]).T
lm_y = np.log(full_data['Voxels'])
lm_fit = LinearRegression().fit(X=lm_X,
                                y=lm_y)
beta0 = lm_fit.intercept_
beta1 = lm_fit.coef_[0]
beta2 = lm_fit.coef_[1]
yhat = lm_fit.predict(lm_X)
sigma = np.sqrt(np.sum(np.square(lm_y - yhat))/(lm_X.shape[0] - 3))
pod_est = calc_pod(pod_data['a'], 100, beta0, beta1, beta2, sigma)

n_boot = 5000
pod_boot = np.zeros((n_boot, pod_data.shape[0]))
for i in range(n_boot):
    lm_y_star = yhat + np.random.normal(0, sigma, yhat.shape)
    lm_fit_star = LinearRegression().fit(X=lm_X, y=lm_y_star)
    beta0_star = lm_fit_star.intercept_
    beta1_star = lm_fit_star.coef_[0]
    beta2_star = lm_fit_star.coef_[1]
    yhat_star = lm_fit_star.predict(lm_X)
    sigma_star = np.sqrt(np.sum(np.square(lm_y_star - yhat_star))/(lm_X.shape[0] - 3))
    pod_est_star = calc_pod(pod_data['a'], 100, beta0_star, beta1_star,
                            beta2_star, sigma_star)
    pod_boot[i, :] = pod_est_star

pod_traditional = pd.DataFrame({'lb': np.quantile(pod_boot, 0.025, axis=0),
                                'ub': np.quantile(pod_boot, 0.975, axis=0),
                                'est': pod_est})

fig, ax = plt.subplots()
ax.plot(pod_data['a'], pod_data['est'], 'r-', label='Eq. 2')
ax.fill_between(pod_data['a'], pod_data['lb'], pod_data['ub'], color='red', alpha=0.25)
ax.plot(pod_data['a'], pod_traditional['est'], 'k--', label='Conventional')
ax.fill_between(pod_data['a'], pod_traditional['lb'],
                pod_traditional['ub'], color='black', alpha=0.25)
ax.set_xscale('log')
ax.set_xlabel('$a$ (mm$^3$)')
ax.set_ylabel('POD')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()

plt.savefig('no_sim_results_compare_quad.png', dpi=600)
