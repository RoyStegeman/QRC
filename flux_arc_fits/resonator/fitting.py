import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.optimize import curve_fit

    INLIER_THRESHOLD = 0.2*6  # approximate width of a peak in the qubit spectroscopy in Hz
    return INLIER_THRESHOLD, Path, curve_fit, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load data
    """)
    return


@app.cell
def _(Path, np):
    current_dir = Path(__file__).resolve().parent
    data = np.load(current_dir / "resonator_data_10.npz")
    return (data,)


@app.cell
def _(data, np):
    # This is a single-qubit experiment so just pick the only qubit
    qubit = list(data.keys())[0]
    data_arr = data[qubit]
    freq_data = data_arr["freq"]
    bias_data = data_arr["bias"]
    signal_data = data_arr["signal"]

    freq = np.unique(freq_data)
    bias = np.unique(bias_data)

    signal = signal_data.reshape(len(bias), len(freq))
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
    ### Find peaks
    """)
    return


@app.cell
def _(bias, freq, np, signal):
    from scipy.signal import find_peaks
    from scipy.ndimage import median_filter, gaussian_filter1d
    from scipy.special import erfinv
    from qibocal.protocols.flux_dependence import utils


    median_per_flux = np.median(signal, axis=1, keepdims=True)
    median_per_frequency = np.median(signal, axis=0, keepdims=True)
    global_median = np.median(signal)

    # Match the flux feature extraction preprocessing to reduce background
    # gradients and normalize each bias row before peak picking.
    filtered_signal = utils.filter_data(signal)
    scaled_signal = utils.minmax_scaling(filtered_signal, axis=1)

    # Sometimes there are bright spots for a given bias. Not sure what causes them,
    # but this should get rid of them.
    median_per_bias = np.median(scaled_signal, axis=1, keepdims=True)
    centered_signal = scaled_signal - median_per_bias

    bias_pts, freq_pts = [], []
    is_peak = []
    for bias_val, row in zip(bias, centered_signal):
        # Detect both peaks and dips explicitly and keep the most prominent
        # feature per bias row.
        peaks, peak_props = find_peaks(row, prominence=0.2)
        dips, dip_props = find_peaks(-row, prominence=0.2)
        if len(peaks) == 0 and len(dips) == 0:
            continue

        if len(dips) == 0 or (
            len(peaks) > 0
            and peak_props["prominences"].max() >= dip_props["prominences"].max()
        ):
            best = peaks[np.argmax(peak_props["prominences"])]
            best_is_peak = True
        else:
            best = dips[np.argmax(dip_props["prominences"])]
            best_is_peak = False

        bias_pts.append(bias_val)
        freq_pts.append(freq[best])
        is_peak.append(best_is_peak)


    # Keep only the dominant extremum type to reject rows detecting the opposite feature.
    select_peaks = sum(is_peak) >= (len(is_peak) / 2)
    mask = np.equal(is_peak, select_peaks)

    bias_pts = np.asarray(bias_pts)[mask]
    freq_pts = np.asarray(freq_pts)[mask]
    return bias_pts, freq_pts


@app.cell
def _(bias, bias_pts, freq, freq_pts, plt, signal_residuals):
    plt.pcolormesh(freq, bias, signal_residuals, cmap="viridis")
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
def _(np):
    def G_f_d(xi, xj, offset, d, crosstalk_element, normalization):

        return (
            d**2
            + (1 - d**2)
            * np.cos(
                np.pi
                * (xi * normalization + normalization * xj * crosstalk_element + offset)
            )
            ** 2
        ) ** 0.25

    def transmon_frequency(
        xi, xj, w_max, d, normalization, offset, crosstalk_element, charging_energy
    ):

        return (w_max + charging_energy) * G_f_d(
            xi,
            xj,
            offset=offset,
            d=d,
            normalization=normalization,
            crosstalk_element=crosstalk_element,
        ) - charging_energy

    def transmon_readout_frequency(
        xi,
        xj,
        w_max,
        d,
        normalization,
        crosstalk_element,
        offset,
        resonator_freq,
        g,
        charging_energy,
    ):

        qubit_frequency = transmon_frequency(
            xi=xi,
            xj=xj,
            w_max=w_max,
            d=d,
            normalization=normalization,
            offset=offset,
            crosstalk_element=crosstalk_element,
            charging_energy=charging_energy,
        )
        return resonator_freq + g**2 * (
            1 / (resonator_freq - qubit_frequency)
            - 1 / (resonator_freq - qubit_frequency + charging_energy)
        )


    def fit_function(
        x: float,
        g: float,
        d: float,
        offset: float,
        normalization: float,
        freq: float,
        charging_energy: float,
    ):
        """Fit function for resonator flux dependence."""
        return transmon_readout_frequency(
            xi=x,
            w_max=5.5e9 * 1e-9, # TODO: this is loaded from calibration file?
            xj=0,
            d=d,
            normalization=normalization,
            offset=offset,
            crosstalk_element=1,
            charging_energy=charging_energy,
            resonator_freq=freq,
            g=g,
        )

    return (fit_function,)


@app.cell
def _(INLIER_THRESHOLD, bias_pts, curve_fit, fit_function, freq_pts, np):


    freq_ghz = freq_pts / 1e9

    p0 = np.array([
        0.1, # g
        0.1, #d
        0.0, # phase offset
        1.0, # normalization
        np.mean(freq_ghz), # resonator frequency
        0.2, # charging energy
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
            popt, _ = curve_fit(
                fit_function,
                bias_pts[subset],
                freq_ghz[subset],
                p0=p0,
                method="lm",  # lm is a fast option
            )
        except RuntimeError:
            # failed to converge on this subset
            continue


        residuals_all = np.abs(freq_ghz - fit_function(bias_pts, *popt))
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
    return best_inliers, best_params, freq_ghz


@app.cell
def _(best_inliers, best_params, bias_pts, curve_fit, fit_function, freq_ghz):
    # Finally optimize by doing a least-squares fit to the best set of inliers
    final_params, _ = curve_fit(
        fit_function,
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
def _(
    best_inliers,
    best_params,
    bias,
    bias_pts,
    final_params,
    fit_function,
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
    freq_plot_vals = fit_function(bias_plot_vals, *final_params) * 1e9
    plt.plot(freq_plot_vals, bias_plot_vals, color='white', label='Fit')
    freq_plot_vals_best_params = fit_function(bias_plot_vals, *best_params) * 1e9
    plt.plot(freq_plot_vals_best_params, bias_plot_vals, color='red', ls='--', label='prior')

    plt.xlim(freq_data.min(), freq_data.max())
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Bias [a.u.]")
    plt.colorbar(label="Signal [a.u.]")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
