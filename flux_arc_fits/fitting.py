import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks


    return find_peaks, np, plt


@app.cell
def _(np):
    data = np.load("./qubit_data_0.npz")
    return (data,)


@app.cell
def _(data, np):
    qubit = list(data.keys())[0] # assume a single relevant qubit
    arr = data[qubit]
    freq_data = arr["freq"]
    bias_data = arr["bias"]
    signal_data = arr["signal"]

    freq = np.unique(freq_data)
    bias = np.unique(bias_data)
    signal = signal_data.reshape(len(bias), len(freq))
    return bias, freq, signal


@app.cell
def _(bias, freq, plt, signal):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(freq, bias, signal, shading="nearest", cmap="viridis")
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.show()
    return


@app.cell
def _(bias, freq, np, plt, signal):
    # the frequency dependence of the resonator is flux-dependent, but the frequency
    # dependence of the elements that affect the background signal (cable response,
    # amplifiers) do not depend on flux. Therefore subtracting the mean across a frequency
    # bin is a good way to estimate the non-resonator contribution to the background.
    # NOTE: this works on the assumption that the resonator peak does not affect the median
    median_per_freq = np.median(signal, axis=0, keepdims=True)
    signal_without_median = (signal - median_per_freq) # minus in front so we can look for peaks
    plt.pcolormesh(freq, bias, signal_without_median, shading="nearest", cmap="viridis")
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.show()
    return (signal_without_median,)


@app.cell
def _(bias, find_peaks, freq, np, signal_without_median):
    prominence = np.median(np.abs(signal_without_median - np.median(signal_without_median)))
    bias_pts, freq_pts, amp_pts = [], [], []
    for i, row in enumerate(signal_without_median):
        # NOTE: the prominence choice is important!
        peaks, props = find_peaks(row, prominence=prominence) # use find_peaks instead of argmax because there may be nothing in a row
        if len(peaks) == 0:
            continue
        # keep the strongest candidate in this row
        best = peaks[np.argmax(props["prominences"])]
        bias_pts.append(bias[i])
        freq_pts.append(freq[best])
        amp_pts.append(row[best])
    return bias_pts, freq_pts


@app.cell
def _(bias, bias_pts, freq, freq_pts, plt, signal_without_median):
    plt.pcolormesh(freq, bias, signal_without_median, shading="nearest", cmap="viridis")
    plt.scatter(freq_pts, bias_pts, color='white', marker='.')
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.colorbar(label="Signal")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
