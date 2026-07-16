import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load data
    """)
    return


@app.cell
def _(np):
    data = np.load("./qubit_data_5.npz")
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
    return bias, freq, signal


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
    ### Fit parabola
    """)
    return


@app.cell
def _(bias_pts, freq_pts, np):
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    # residual_threshold is 1e6 as this is the approximate with of a qubit lorentzian
    ransac_parabola = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        RANSACRegressor(min_samples=3, residual_threshold=1e6, random_state=0)
    )

    ransac_parabola.fit(np.array(bias_pts).reshape(-1, 1), np.array(freq_pts));
    return (ransac_parabola,)


@app.cell
def _(
    bias,
    bias_pts,
    filtered_signal,
    freq,
    freq_pts,
    np,
    plt,
    ransac_parabola,
):
    plt.pcolormesh(freq, bias, filtered_signal, cmap="viridis")
    plt.scatter(freq_pts, bias_pts, color='white', marker='.', label='Detected Peaks')

    # Plot the RANSAC-fitted parabola
    bias_plot_vals = np.linspace(bias.min(), bias.max(), 500).reshape(-1, 1)
    freq_plot_vals = ransac_parabola.predict(bias_plot_vals)
    plt.plot(freq_plot_vals, bias_plot_vals, color='red', label='Fit')

    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
