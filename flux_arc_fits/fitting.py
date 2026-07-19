import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, mo, np, plt


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
    data = np.load(current_dir / "qubit_data_0.npz")
    return (data,)


@app.cell
def _(data, np):
    # this is a single-qubit experiment so just pick the only qubit
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
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove background
    """)
    return


@app.cell
def _(np, signal):
    # the frequency dependence of the resonator is flux-dependent, but the frequency
    # dependence of the elements that affect the background signal (cable response,
    # amplifiers) do not depend on flux. Therefore subtracting the mean across a frequency
    # bin is a good way to estimate the non-resonator contribution to the background.
    #
    # NOTE: this works on the assumption that the resonator peak does not affect the median
    col_median = np.median(signal, axis=0, keepdims=True)
    col_diff = signal - col_median

    # Subtract also the row median, to remove the background contribution we see as the
    # gradient centred on the peak of the flux-arc.
    #
    # NOTE: this is probably a consequence of reading the resonator at the flux bias point,
    # so can be avoided by moving the readout frequency to follow the resonator flux-arc
    row_median = np.median(col_diff, axis=1, keepdims=True)
    double_diff = col_diff - row_median

    # Determine if the remaining feature is a peak or a dip
    sign = 1 if np.abs(double_diff.max()) > np.abs(double_diff.min()) else -1
    filtered_signal = sign * double_diff
    return (filtered_signal,)


@app.cell
def _(bias, filtered_signal, freq, plt):
    plt.pcolormesh(freq, bias, filtered_signal, cmap="viridis")
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
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

    prominence = np.median(np.abs(filtered_signal - np.median(filtered_signal)))
    bias_pts, freq_pts, amp_pts = [], [], []
    for index_filtered_signal, row in enumerate(filtered_signal):
        smoothed_row = gaussian_filter1d(row, sigma=2)
        row_mad = np.median(np.abs(smoothed_row - np.median(smoothed_row)))
        peaks, props = find_peaks(smoothed_row, prominence=row_mad) # use find_peaks instead of argmax because there may be nothing in a row
        if len(peaks) == 0:
            continue
        # keep the strongest candidate in this row
        best = peaks[np.argmax(props["prominences"])]
        bias_pts.append(bias[index_filtered_signal])
        freq_pts.append(freq[best])

    bias_pts = np.asarray(bias_pts)
    freq_pts = np.asarray(freq_pts)
    return bias_pts, freq_pts


@app.cell
def _(bias, bias_pts, filtered_signal, freq, freq_pts, plt):
    plt.pcolormesh(freq, bias, filtered_signal, cmap="viridis")
    plt.scatter(freq_pts, bias_pts, color='white', marker='.', label='Detected Peaks')

    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Perform the fit
    """)
    return


@app.cell
def _(bias_pts, freq_pts, np):
    from scipy.optimize import root, curve_fit

    def f01_model(bias, EJ1, EJ2, EC, bias_flux_ratio):
        x = np.pi * bias / bias_flux_ratio
        d = (EJ1 - EJ2) / (EJ1 + EJ2)

        EJ = (EJ1 + EJ2) * np.sqrt(
            np.cos(x)**2 + d**2 * np.sin(x)**2
        )

        return np.sqrt(8 * EC * EJ) - EC


    def residuals(log_params, bias, freq):
        EJ1, EJ2, EC = np.exp(log_params[:3])
        bias_flux_ratio = log_params[3]

        return (
            f01_model(
                bias,
                EJ1,
                EJ2,
                EC,
                bias_flux_ratio,
            )
            - freq
        )


    freq_ghz = freq_pts / 1e9

    p0 = np.array([
        np.log(5.5),
        np.log(5.5),
        np.log(0.2),
        np.mean(bias_pts),
    ])


    p_success = 0.999      # desired probability of finding a clean sample
    n_sample = len(p0)
    min_iters = 20
    max_iters = 5000

    best_inliers = None
    best_params = None
    tried = set()

    N_needed = max_iters   # pessimistic until we have an estimate
    ransac_iterations = 0


    while ransac_iterations < min(N_needed, max_iters) or ransac_iterations < min_iters:
        ransac_iterations += 1

        subset = np.random.choice(
            len(bias_pts),
            len(p0),
            replace=False,
        )

        subset_ = tuple(sorted(subset))

        if subset_ in tried:
            continue

        tried.add(subset_)

        result = root(
            residuals,
            p0,
            method="lm",
            args=(
                bias_pts[subset],
                freq_ghz[subset],
            ),
        )

        if not result.success:
            continue

        popt = np.array([
            np.exp(result.x[0]),
            np.exp(result.x[1]),
            np.exp(result.x[2]),
            result.x[3],
        ])

        if not np.all(np.isfinite(popt)):
            continue

        residuals_all = np.abs(
            freq_ghz - f01_model(bias_pts, *popt)
        )

        inliers = residuals_all < 0.5e6 / 1e9

        if ( best_inliers is None or inliers.sum() > best_inliers.sum() ):
            best_inliers = inliers
            best_params = popt

            w_hat = max(best_inliers.sum() / len(bias_pts), 1e-3)   # avoid log(0)
            denom = np.log(1 - w_hat**n_sample)
            if denom < 0:  # guard: w_hat**n_sample < 1
                N_needed = np.log(1 - p_success) / denom

    print(ransac_iterations)
    return best_inliers, best_params, curve_fit, f01_model, freq_ghz, popt


@app.cell
def _(
    best_inliers,
    best_params,
    bias_pts,
    curve_fit,
    f01_model,
    freq_ghz,
    np,
    popt,
):
    # Final least-squares fit using all inliers
    final_params, _ = curve_fit(
        f01_model,
        bias_pts[best_inliers],
        freq_ghz[best_inliers],
        p0=best_params,
        # bounds=(
        #     [0, 0, 0, -np.inf],
        #     [np.inf, np.inf, np.inf, np.inf],
        # ),
        method='lm',
        maxfev=100000,
    )

    if not np.all(final_params > 0):
        print("Warning: LM fit produced nonphysical (negative) EJ/EC:", popt)
    return (final_params,)


@app.cell
def _(final_params):
    # NOTE: the parameters are non-physical and the fit above complains that the covariance could not be estimated. This suggests a degeneracy between parameters (also indicated by the tiny EC). I suspect the degeneracy is becuase we are very zoomed in and therefore don't need all paramters to describe the parabolic shape in the data window.
    final_params
    return


@app.cell
def _(
    best_inliers,
    best_params,
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
    plt.scatter(freq_pts[~best_inliers], bias_pts[~best_inliers], color='white', marker='.', s=60, zorder=10, label='Detected Peaks')
    plt.scatter(freq_pts[best_inliers], bias_pts[best_inliers], color='magenta', marker='.', s=60, zorder=10, label='inliers')
    bias_plot_vals = np.linspace(bias.min(), bias.max(), num=500)
    freq_plot_vals = f01_model(bias_plot_vals, *final_params) * 1e9
    plt.plot(freq_plot_vals, bias_plot_vals, color='black', label='Fit')

    best_params_plot_vals = f01_model(bias_plot_vals, *best_params) * 1e9
    plt.plot(best_params_plot_vals, bias_plot_vals, color='red', label='best_params')

    plt.xlim(freq_data.min(), freq_data.max())
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
