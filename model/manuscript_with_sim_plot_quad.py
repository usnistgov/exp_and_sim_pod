import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

post_samps = az.from_netcdf('./post_samps_with_sim_neg_binom_quad.nc')
post_samps_no_sim = az.from_netcdf('./post_samps_no_sim_neg_binom_quad.nc')

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

est_bounds = pd.DataFrame()
est_bounds['lb'] = np.quantile(post_samps.posterior.voxels_pred, q=0.025, axis=(0, 1))
est_bounds['ub'] = np.quantile(post_samps.posterior.voxels_pred, q=0.975, axis=(0, 1))
est_bounds['Voxels'] = np.quantile(post_samps.posterior.voxels_pred, q=0.5, axis=(0, 1))
est_bounds['a'] = np.exp(np.linspace(np.log(0.0001), np.log(0.15), 200))

est_bounds_no_sim = pd.DataFrame()
est_bounds_no_sim['lb'] = np.quantile(post_samps_no_sim.posterior.voxels_pred, q=0.025, axis=(0, 1))
est_bounds_no_sim['ub'] = np.quantile(post_samps_no_sim.posterior.voxels_pred, q=0.975, axis=(0, 1))
est_bounds_no_sim['Voxels'] = np.quantile(post_samps_no_sim.posterior.voxels_pred, q=0.5, axis=(0, 1))
est_bounds_no_sim['a'] = np.exp(np.linspace(np.log(0.0001), np.log(0.15), 200))


plt_char=['o', '^', 's']
thresh_labels = ['Min. Threshold', 'Mid.', 'Max.']
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xscale('log')
for i in range(3):
    ax.scatter(x=full_data.a[full_data.thresh==(i+1)],
               y=full_data.Voxels[full_data.thresh==(i+1)],
               marker=plt_char[i], c='grey',
               label=thresh_labels[i])
ax.fill_between(est_bounds.a, est_bounds['lb'], est_bounds['ub'],
                color='red', alpha=0.25)
ax.plot(est_bounds.a, est_bounds.Voxels, 'r-', label='Exp. + Sim.')
ax.plot(est_bounds_no_sim.a, est_bounds_no_sim.Voxels, '-', color='tab:blue',
        label='Exp. Only')
ax.fill_between(est_bounds.a, est_bounds_no_sim['lb'], est_bounds_no_sim['ub'],
                color='tab:blue', alpha=0.25)

ax.legend(frameon=False)
ax.set_xlabel('$\\tilde{a}$ (mm$^3$)')
ax.set_ylabel('Voxel Count')
plt.tight_layout()

plt.savefig('compare_with_sim_no_sim.png', dpi=600)

tmp = pd.merge_asof(full_data.sort_values('a'), est_bounds.sort_values('a'), on='a')
tmp = tmp[['a', 'Voxels_x', 'lb', 'ub']]
tmp['in_interval'] = (tmp['Voxels_x'] > tmp['lb']) & (tmp['Voxels_x'] < tmp['ub'])
print(tmp.mean())
print(tmp[tmp['a'] > 1e-3].mean())

pod_data = pd.DataFrame()
for i in range(4):
    pod_data['est' + str(i)] = np.quantile(post_samps.posterior.prob_thresh[:, :, :, i],
                                           q=0.5, axis=(0, 1))
    pod_data['lb' + str(i)] = np.quantile(post_samps.posterior.prob_thresh[:, :, :, i],
                                          q=0.025, axis=(0, 1))
    pod_data['ub' + str(i)] = np.quantile(post_samps.posterior.prob_thresh[:, :, :, i],
                                          q=0.975, axis=(0, 1))
pod_data['a'] = np.exp(np.linspace(np.log(0.00001), np.log(0.005), 200))

pod_data_no_sim = pd.DataFrame()
for i in range(4):
    pod_data_no_sim['est' + str(i)] = np.quantile(post_samps_no_sim.posterior.prob_thresh[:, :, :, i],
                                                  q=0.5, axis=(0, 1))
    pod_data_no_sim['lb' + str(i)] = np.quantile(post_samps_no_sim.posterior.prob_thresh[:, :, :, i],
                                                 q=0.025, axis=(0, 1))
    pod_data_no_sim['ub' + str(i)] = np.quantile(post_samps_no_sim.posterior.prob_thresh[:, :, :, i],
                                                 q=0.975, axis=(0, 1))
    pod_data_no_sim['a'] = np.exp(np.linspace(np.log(0.00001), np.log(0.005), 200))

Vc = [50, 100, 200, 300]
fig, ax = plt.subplots(nrows=4)
for i in range(4):
    ax[i].plot(pod_data['a'], pod_data['est' + str(i)], 'r-', label='Exp. + Sim.')
    ax[i].fill_between(pod_data['a'], pod_data['lb' + str(i)], pod_data['ub' + str(i)],
                       color='red', alpha=0.25)
    ax[i].plot(pod_data['a'], pod_data_no_sim['est' + str(i)], '-', color='tab:blue', label='Exp. Only')
    ax[i].fill_between(pod_data['a'], pod_data_no_sim['lb' + str(i)],
                       pod_data_no_sim['ub' + str(i)], color='tab:blue', alpha=0.25)
    ax[i].set_xscale('log')
    ax[i].set_xlabel('$a$ (mm$^3$)')
    ax[i].set_ylabel('POD')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].legend(frameon=False)
    ax[i].set_xlim([1e-4, 5e-3])
    ax[i].annotate('$V_c$ = ' + str(Vc[i]), (0.05, 0.75), xycoords='axes fraction')
plt.tight_layout()

plt.savefig('pod_compare_with_sim_no_sim.png', dpi=600)
