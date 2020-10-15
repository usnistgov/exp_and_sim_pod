## Prerequisites

Python must be installed
[https://www.r-project.org/](https://www.r-project.org/) as well as
the Python packages 

## Running the code

You should be able to run the `.R` files interactively or in BATCH
mode.

## List of files and descriptions

- `data/`
    1. `a^_vs_a_image_threshold_v2.xlsx`
	2. `simulation0.csv`
	3. `simulation1.csv`
	4. `simulation2.csv`
	5. `simulation3.csv`
	6. `simulation4.csv`
	8. `simulation5.csv`
	9. `simulation6.csv`
	10. `simulation7.csv`
	11. `simulation8.csv`
	12. `simulation9.csv`
	13. `simulation10.csv`
	14. `simulation11.csv`
- `model/`
    1. `model_no_sim_neg_binom_quad.stan`
    2. `model_with_wim_neg_binom_quad.stan`
    3. `run_model_no_sim_neg_binom_quad.py`
        - *input* -- `a^_vs_a_image_threshold_v2.xlsx` and
          `model_no_sim_neg_binom_quad.stan`
        - *output* -- `post_samps_no_som_neg_binom_quad.nc`
    4. `run_model_with_sim_neg_binom_quad.py`
        - *input* -- `a^_vs_a_image_threshold_v2.xlsx`,
          `simulation*.csv`, and `model_with_sim_neg_binom_quad.stan`
        - *output* -- `post_samps_iwth_sim_neg_binom_quad.nc`
    5. `manuscript_no_sim_plot_quad.py`
        - *input* -- `a^_vs_a_image_threshold_v2.xlsx` and
          `post_samps_no_sim_neg_binom_quad.nc`
        - *output* -- `no_sim_results_manuscript_neg_binom_quad.png`
          and `no_sim_results_compare_quad.png`
    6. `manuscript_with_sim_plot_quad.py`
        - *input* -- `a^_vs_a_image_threshold_v2.xlsx`,
          `simulation*.csv`, `post_samps_with_sim_neg_binom_quad.nc`,
          and `post_samps_no_sim_neg_binom_quad.nc`
        - *output* -- `compare_with_sim_no_sim.png` and
          `pod_compare_with_sim_no_sim.png`

