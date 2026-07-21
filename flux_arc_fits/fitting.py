import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.optimize import curve_fit


    return Path, curve_fit, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We should allow for multiple fit functions since fitting the full f01 function is quite time consuming and in most cases the user is probably just interested in knowing the peak of the flux arc, which can be achieved much faster by a simple 2nd or 4th order polynomial.

    Should d=0 also be toggleable?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load data
    """)
    return


@app.cell
def _(Path, np):
    current_dir = Path(__file__).resolve().parent
    data = np.load(current_dir / "qubit_data_3.npz")
    return (data,)


@app.cell
def _(data, np):
    # This is a single-qubit experiment so just pick the only qubit
    qubit = list(data.keys())[0]
    data_arr = data[qubit]
    freq_data = data_arr["freq"]
    bias_data = data_arr["bias"]
    signal_data = data_arr["signal"]

    freq, freq_idx = np.unique(freq_data, return_inverse=True)
    bias, bias_idx = np.unique(bias_data, return_inverse=True)

    signal = np.full((len(bias), len(freq)), np.nan)
    signal[bias_idx, freq_idx] = signal_data
    return bias, freq, freq_data, signal


@app.cell
def _(bias, freq, plt, signal):
    plt.pcolormesh(freq, bias, signal, cmap="viridis")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Bias [a.u.]")
    plt.colorbar(label="Signal [a.u.]")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove background (along column is just for the resonator)
    """)
    return


@app.cell
def _(np, signal):
    # The frequency dependence of the resonator is flux-dependent, but the frequency
    # dependence of the elements that affect the background signal (cable response,
    # amplifiers) do not depend on flux. Therefore subtracting the mean across a frequency
    # bin is a good way to estimate the non-resonator contribution to the background.
    #
    # NOTE: this works on the assumption that the resonator peak does not affect the median
    # col_median = np.median(signal, axis=0, keepdims=True)
    # col_diff = signal - col_median

    # Subtract also the row median, to remove the background contribution we see as the
    # gradient centred on the peak of the flux-arc.
    #
    # NOTE: this is probably a consequence of reading the resonator at the flux bias point,
    # so can be avoided by moving the readout frequency to follow the resonator flux-arc
    row_median = np.median(signal, axis=1, keepdims=True)
    double_diff = signal - row_median

    # Determine if the remaining feature is a peak or a dip
    sign = 1 if np.abs(double_diff.max()) > np.abs(double_diff.min()) else -1
    filtered_signal = sign * double_diff
    return (filtered_signal,)


@app.cell
def _(bias, filtered_signal, freq, plt):
    plt.pcolormesh(freq, bias, filtered_signal, cmap="viridis")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Bias [a.u.]")
    plt.colorbar(label="Signal [a.u.]")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Find peaks
    """)
    return


@app.cell
def _(bias, filtered_signal, freq, np):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.special import erfinv

    bias_pts, freq_pts = [], []
    for index_filtered_signal, row in enumerate(filtered_signal):
        # The Gaussian filter not only reduces noise in the background far away from the
        # arc, but also reduces noise within the arc, which may result in peaks being
        # detected correctly that otherwise would have been missed (see e.g
        # qubit_data_3.npz).
        smoothed_row = gaussian_filter1d(row, sigma=2)
        # The standard deviation is computed from the median absolute deviation instead of
        # the standard deviation itself to avoid the peaks in the arc from affecting the
        # estimate of the background noise. While this prominence threshold is somewhat
        # motivated, it is still a choice and it has been observed that the result is not
        # very sensitive to it and probably it is even fine to set the threshold to 0.
        row_mad = np.median(np.abs(smoothed_row - np.median(smoothed_row)))
        row_std = 1.0 / (np.sqrt(2) * erfinv(0.5)) * row_mad
        # Use find_peaks instead of argmax because there may be nothing in a row
        peaks, props = find_peaks(smoothed_row, prominence=row_std)

        if len(peaks) == 0:
            continue

        # Keep only the peak with the largest prominence per bias
        best = peaks[np.argmax(props["prominences"])]
        bias_pts.append(bias[index_filtered_signal])
        freq_pts.append(freq[best])

    bias_pts = np.asarray(bias_pts)
    freq_pts = np.asarray(freq_pts)
    return bias_pts, freq_pts


@app.cell
def _(bias, bias_pts, filtered_signal, freq, freq_pts, plt):
    plt.pcolormesh(freq, bias, filtered_signal, cmap="viridis")
    plt.scatter(freq_pts, bias_pts, color='white', marker='.', s=60, zorder=10, label='Detected peaks')

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Bias [a.u.]")
    plt.colorbar(label="Signal [a.u.]")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Perform the fit using RANSAC
    """)
    return


@app.cell
def _(bias_pts, curve_fit, freq_pts, np):
    # from scipy.optimize import curve_fit

    INLIER_THRESHOLD = 0.5e6  # approximate width of a peak in the qubit spectroscopy in Hz

    def f01_model(bias, EJ1, EJ2, EC, bias_flux_ratio):
        """Eq. 14.38 from Manenti & Motta"""
        x = np.pi * bias / bias_flux_ratio
        d = (EJ1 - EJ2) / (EJ1 + EJ2)
        EJ = (EJ1 + EJ2) * np.sqrt(np.cos(x)**2 + d**2 * np.sin(x)**2)
        return np.sqrt(8 * EC * EJ) - EC


    def f01_model_logparams(bias, log_EJ1, log_EJ2, log_EC, bias_flux_ratio):
        # We fit the exponent because it has been observed that the fitted energies take
        # unphysical values with EJ1 and EJ2 becoming many orders of magnitude larger and
        # EC many orders of magnitude smaller than physical. This slows down the fit, but
        # taking the exponent speeds up fits across many orders of magnitude.
        EJ1, EJ2, EC = np.exp([log_EJ1, log_EJ2, log_EC])
        return f01_model(bias, EJ1, EJ2, EC, bias_flux_ratio)


    freq_ghz = freq_pts / 1e9

    p0 = np.array([
        np.log(5.5),
        np.log(5.5),
        np.log(0.2),
        np.mean(bias_pts),
    ])


    # The number of iterations is determined following the standard for RANSAC
    # https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
    p_success = 0.999  # desired probability of finding a sample containing only inliers
    min_iters = 100
    max_iters = 5000


    N_needed = np.inf
    ransac_iterations = 0
    best_inliers = np.array([])
    best_params = np.array([])
    tried_subsets = set()
    while ransac_iterations < min(N_needed, max_iters) or ransac_iterations < min_iters:
        ransac_iterations += 1

        subset = np.random.choice(len(bias_pts), len(p0), replace=False)
        subset_ = tuple(sorted(subset))
        if subset_ in tried_subsets:
            continue
        tried_subsets.add(subset_)

        try:
            popt_log, _ = curve_fit(
                f01_model_logparams,
                bias_pts[subset],
                freq_ghz[subset],
                p0=p0,
                method="lm",  # lm is a fast option
            )
        except RuntimeError:
            # failed to converge on this subset
            continue

        popt = np.array([
            np.exp(popt_log[0]),
            np.exp(popt_log[1]),
            np.exp(popt_log[2]),
            popt_log[3],
        ])

        residuals_all = np.abs(freq_ghz - f01_model(bias_pts, *popt))
        inliers = residuals_all < INLIER_THRESHOLD / 1e9
        if inliers.sum() == len(bias_pts):
            # all points are inliers, so we can proceed
            best_inliers = inliers
            best_params = popt
            break

        if inliers.sum() >= len(subset) and inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_params = popt
            denom = np.log(1 - (best_inliers.sum() / len(bias_pts))**len(subset))
            N_needed = np.log(1 - p_success) / denom
    return best_inliers, best_params, f01_model, freq_ghz


@app.cell
def _(best_inliers, best_params, bias_pts, curve_fit, f01_model, freq_ghz):
    # Finally optimize by doing a least-squares fit to the best set of inliers
    final_params, _ = curve_fit(
        f01_model,
        bias_pts[best_inliers],
        freq_ghz[best_inliers],
        p0=best_params,
        method='lm',
        maxfev=100000,
    )
    return (final_params,)


@app.cell
def _(final_params):
    # NOTE: the parameters are non-physical and the fit above complains that the covariance
    # could not be estimated. This suggests a degeneracy between parameters (also indicated
    # by the tiny EC). I suspect the degeneracy is because we are very zoomed in and
    # therefore don't need all parameters to describe the parabolic shape in the data window.
    final_params
    return


@app.cell
def _(bias, final_params, np):
    EJ1, EJ2, EC, bias_flux_ratio = final_params

    # Find integers k that fall within the experimental bias range
    bias_min, bias_max = bias.min(), bias.max()
    k_start = int(np.ceil(min(bias_min / bias_flux_ratio, bias_max / bias_flux_ratio)))
    k_end = int(np.floor(max(bias_min / bias_flux_ratio, bias_max / bias_flux_ratio)))
    k_values = np.arange(k_start, k_end + 1)

    peak_biases = k_values * bias_flux_ratio
    peak_biases = peak_biases[(peak_biases >= bias_min) & (peak_biases <= bias_max)]

    # Select the peak closest to 0 bias and compute its frequency
    if len(peak_biases) > 0:
        best_point_bias = peak_biases[np.argmin(np.abs(peak_biases))]
        # At the absolute maximum, EJ simplifies cleanly to (EJ1 + EJ2)
        best_point_freq_ghz = np.sqrt(8 * EC * (EJ1 + EJ2)) - EC
        best_point_freq = best_point_freq_ghz * 1e9
    else:
        best_point_bias = None
    return best_point_bias, best_point_freq


@app.cell
def _(
    best_inliers,
    best_point_bias,
    best_point_freq,
    bias,
    bias_pts,
    f01_model,
    final_params,
    freq,
    freq_data,
    freq_pts,
    np,
    plt,
    signal,
):
    plt.pcolormesh(freq, bias, signal, cmap="viridis")
    plt.scatter(freq_pts[~best_inliers], bias_pts[~best_inliers], color='lime', marker='.', s=60, zorder=10, label='Outliers')
    plt.scatter(freq_pts[best_inliers], bias_pts[best_inliers], color='white', marker='.', s=60, zorder=10, label='Inliers')
    bias_plot_vals = np.linspace(bias.min(), bias.max(), num=500)
    freq_plot_vals = f01_model(bias_plot_vals, *final_params) * 1e9
    plt.plot(freq_plot_vals, bias_plot_vals, color='white', label='Fit')

    if best_point_bias is not None:
        plt.scatter( best_point_freq, best_point_bias, color='red', marker='.', s=60, zorder=15, label="Best point")

    plt.xlim(freq_data.min(), freq_data.max())
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Bias [a.u.]")
    plt.colorbar(label="Signal [a.u.]")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
