import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def plot_experiment_comparison(experiment_names, base_output_path):
    """
    Load outputs from multiple HOWFSC experiments and produce comparison plots.

    For each experiment, loads ``measured_contrast.csv``, ``debugging_history.csv``,
    and ``config.yml`` from ``<base_output_path>/<experiment_name>/``. Produces three
    plots saved to ``base_output_path``:

    - ``comparison_contrast_vs_iteration.pdf``: measured contrast vs. iteration number
      for all experiments.
    - ``comparison_contrast_vs_time.pdf``: measured contrast vs. cumulative iteration
      time in minutes.
    - ``comparison_contrast_vs_time_overhead.pdf``: measured contrast vs. cumulative
      iteration time including 60 minutes of additional overhead per iteration.

    Legend entries include the estimator class, noise-free flag, regularization (beta)
    at iteration 1, and the maximum beta value across all iterations.

    Parameters
    ----------
    experiment_names : list of str
        List of experiment folder names, e.g.
        ``['2026-03-26_154159_corgisim_model', '2026-03-31_144413_corgisim_model']``.
        Each must be a subdirectory of ``base_output_path``.
    base_output_path : str
        Path to the directory containing all experiment folders. Comparison
        plots are saved here.
    """

    fig_iter, ax_iter = plt.subplots(layout='constrained')
    fig_time, ax_time = plt.subplots(layout='constrained')
    fig_time_oh, ax_time_oh = plt.subplots(layout='constrained')

    for name in experiment_names:
        experiment_path = os.path.join(base_output_path, name)

        # Load measured contrast
        measured_c = np.loadtxt(
            os.path.join(experiment_path, 'measured_contrast.csv'),
            delimiter=',', skiprows=1
        )

        # Load debugging history
        debug_df = pd.read_csv(os.path.join(experiment_path, 'debugging_history.csv'))

        # Load config
        with open(os.path.join(experiment_path, 'config.yml'), 'r') as f:
            config = yaml.safe_load(f)

        # Extract legend metadata
        estimator = config.get('objects', {}).get('estimator_class', 'Unknown')
        is_noise_free = config.get('corgi_overrides', {}).get('is_noise_free', 'N/A')

        # Beta: one value per iteration (same across wavelengths), take lam_index == 0
        beta_per_iter = debug_df[debug_df['lam_index'] == 0]['beta'].values
        beta_iter1 = beta_per_iter[0] if len(beta_per_iter) > 0 else float('nan')
        beta_max = np.max(beta_per_iter) if len(beta_per_iter) > 0 else float('nan')

        label = (
                f'estimator={estimator}\nnoise_free={is_noise_free}\n'
                + r'$\beta_0$' + f'={beta_iter1:.1f}\n'
                + r'$\beta_{max}$' + f'={beta_max:.1f}'
        )

        # Contrast vs iteration
        iterations = np.arange(1, len(measured_c) + 1)
        ax_iter.plot(iterations, measured_c, marker='o', label=label)

        # Contrast vs time
        if 'this_iter_dur' in debug_df.columns:
            durations = debug_df[debug_df['lam_index'] == 0]['this_iter_dur'].values  # seconds
            cumulative_time = np.cumsum(durations) / 60  # convert to minutes
            ax_time.plot(cumulative_time, measured_c, marker='o', label=label)

            # Contrast vs time with overhead
            cumulative_time_with_overhead = cumulative_time + np.arange(1, len(durations) + 1) * 60
            ax_time_oh.plot(cumulative_time_with_overhead, measured_c, marker='o', label=label)

    # Contrast vs iteration
    ax_iter.set_xlabel('Iteration')
    ax_iter.set_ylabel('Measured Contrast')
    ax_iter.semilogy()
    ax_iter.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig_iter.savefig(
        os.path.join(base_output_path, 'comparison_contrast_vs_iteration.pdf'),
        bbox_inches='tight'
    )
    plt.close(fig_iter)

    # Contrast vs time
    ax_time.set_xlabel('Spacecraft Time [minutes]')
    ax_time.set_ylabel('Measured Contrast')
    ax_time.semilogy()
    ax_time.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig_time.savefig(
        os.path.join(base_output_path, 'comparison_contrast_vs_time.pdf'),
        bbox_inches='tight'
    )
    plt.close(fig_time)

    # Contrast vs time with overhead
    ax_time_oh.set_xlabel('GITL Time (w/comms overhead) [minutes]')
    ax_time_oh.set_ylabel('Measured Contrast')
    ax_time_oh.semilogy()
    ax_time_oh.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig_time_oh.savefig(
        os.path.join(base_output_path, 'comparison_contrast_vs_time_overhead.pdf'),
        bbox_inches='tight'
    )
    plt.close(fig_time_oh)