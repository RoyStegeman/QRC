import marimo

__generated_with = "0.23.14"
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
    ### Load data
    """)
    return


@app.cell
def _(Path, np):
    current_dir = Path(__file__).resolve().parent
    data = np.load(current_dir / "qubit_data_8.npz")
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
    for i, row in enumerate(filtered_signal):
        smoothed_row = gaussian_filter1d(row, sigma=2)
        row_mad = np.median(np.abs(smoothed_row - np.median(smoothed_row)))
        peaks, props = find_peaks(smoothed_row, prominence=row_mad) # use find_peaks instead of argmax because there may be nothing in a row
        if len(peaks) == 0:
            continue
        # keep the strongest candidate in this row
        best = peaks[np.argmax(props["prominences"])]
        bias_pts.append(bias[i])
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
    from scipy.optimize import curve_fit


    def f01_model(phi, EJ1, EJ2, EC, Phi0):
        """Eq. 14.38 from Manenti, Motta"""
        x = np.pi * phi / Phi0
        d = (EJ1 - EJ2) / (EJ1 + EJ2)
        EJ = (EJ1 + EJ2) * np.sqrt(
            np.cos(x)**2 + d**2 * np.sin(x)**2
        )
        return np.sqrt(8 * EC * EJ) - EC


    freq_ghz = freq_pts / 1e9

    p0 = [
        5.5, # EJ1/h in GHz
        5.5, # EJ2/h in GHz
        0.2, # EC/h in GHz
        np.mean(bias_pts), # Phi0
    ]

    bounds = (
        [0, 0, 0, -np.inf],
        [np.inf, np.inf, np.inf, np.inf]
    )

    best_inliers = None
    best_params = None

    for _ in range(100):

        subset = np.random.choice(len(bias_pts), len(p0), replace=False)

        try:
            popt, _ = curve_fit(
                f01_model,
                bias_pts[subset],
                freq_ghz[subset],
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
        except RuntimeError:
            continue

        residuals = np.abs(freq_ghz - f01_model(bias_pts, *popt))
        inliers = residuals < 0.5e6/1e9

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_params = popt

    # Refit using all inliers
    popt, _ = curve_fit(
        f01_model,
        bias_pts[best_inliers],
        freq_ghz[best_inliers],
        p0=best_params,
        bounds=bounds,
        maxfev=100000
    )
    return f01_model, popt


@app.cell
def _(
    bias,
    bias_pts,
    f01_model,
    filtered_signal,
    freq,
    freq_data,
    freq_pts,
    np,
    plt,
    popt,
):
    plt.pcolormesh(freq, bias, filtered_signal, cmap="viridis")
    plt.scatter(freq_pts, bias_pts, color='white', marker='.', label='Detected Peaks')
    bias_plot_vals = np.linspace(bias.min(), bias.max(), num=500)
    freq_plot_vals = f01_model(bias_plot_vals, *popt) * 1e9
    plt.plot(freq_plot_vals, bias_plot_vals, color='red', label='Fit')

    plt.xlim(freq_data.min(), freq_data.max())
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
