data {
  int N_exp;
  int N_thresh_group;
  int voxels_exp[N_exp];
  vector[N_exp] a_exp;
  vector[N_exp] a_exp_unc;
  vector[N_exp] thresh_exp;
  int thresh_group_exp[N_exp];

  int N_pod;
  vector[N_pod] a_pod;
  int N_voxel_threshold;
  int voxel_threshold[N_voxel_threshold];

  int N_pred;
  vector[N_pred] a_pred;
}

transformed data {
  vector[N_exp] la_exp;
  vector[N_exp] la_exp_unc;
  vector[N_pod] la_pod;
  vector[N_pred] la_pred;

  la_exp = log(a_exp);
  la_exp_unc = a_exp_unc ./ a_exp;
  la_pod = log(a_pod);
  la_pred = log(a_pred);
}

parameters {
  real a_int;
  real a_slope;
  real a_quad;
  vector[N_thresh_group] thresh_slope_eff;
  real thresh_slope_mean;
  real<lower=0> thresh_slope_sd;

  real<lower=0> ovr_disp;

  vector[N_exp] la_exp_true;
}

transformed parameters {
  vector[N_exp] llambda_exp;
  vector[N_thresh_group] thresh_slope;

  thresh_slope = thresh_slope_mean + thresh_slope_sd*thresh_slope_eff;

  llambda_exp = a_int + (a_slope * la_exp_true) +
    (a_quad * (la_exp_true .* la_exp_true)) +
    (thresh_slope[thresh_group_exp] .* thresh_exp);
}

model {
  voxels_exp ~ neg_binomial_2_log(llambda_exp, ovr_disp);
  thresh_slope_eff ~ normal(0.0, 1.0);
  // the average slope is aroun -0.16, so this prior is saying that
  // the individual slopes are probably the same sign as the average
  // thresh_slope_sd ~ normal(0, 0.5*0.16); 

  la_exp_true ~ normal(la_exp, la_exp_unc);
}

generated quantities {
  vector[N_pod] llambda_pod;
  matrix[N_pod, N_voxel_threshold] prob_thresh;
  int voxels_pred[N_pred];
  vector[N_pred] llambda_pred;
  real gen_thresh_slope;
  vector[3] theta;

  theta[1]=1.0/3.0;
  theta[2]=1.0/3.0;
  theta[3]=1.0/3.0;

  gen_thresh_slope = thresh_slope_mean + thresh_slope_sd*normal_rng(0, 1);

  // the multiplication by 1.0 is needed to coerce the int to a real
  llambda_pred = a_int + a_slope*la_pred +
    a_quad*(la_pred .* la_pred) +
    gen_thresh_slope*(categorical_rng(theta) * 1.0 - 2.0);

  voxels_pred = neg_binomial_2_log_rng(llambda_pred, ovr_disp);

  llambda_pod = a_int + a_slope*la_pod +
    a_quad*(la_pod .* la_pod) +
    gen_thresh_slope*(categorical_rng(theta) * 1.0 - 2.0);

  for (i in 1:N_pod) {
    for (j in 1:N_voxel_threshold) {
      prob_thresh[i, j] = 1.0 - neg_binomial_2_cdf(voxel_threshold[j], exp(llambda_pod[i]), ovr_disp);
    }
  }
}
