import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import numpy as np
    from pathlib import Path
    import json
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d


    return Path, json, np, plt, savgol_filter


@app.cell
def _(Path, json, np):
    data_version = 1

    current_dir = Path(__file__).resolve().parent

    npz_file = np.load(current_dir / f"data_{data_version}.npz")
    qubit = list(npz_file.keys())[0]
    signal = npz_file[qubit]

    with open(current_dir/ f"data_{data_version}.json") as file:
        # Parsed straight from the open file object
        data_json = json.load(file)

    amplitudes = np.asarray(data_json['"amplitudes"'])
    frequencies = np.asarray(data_json['"frequencies"'][qubit])
    mag = np.linalg.norm(signal,axis=-1)
    return amplitudes, frequencies, mag


@app.cell
def _(amplitudes, frequencies, mag, plt):
    plt.figure(figsize=(14, 5))   # (width, height) in inches
    plt.pcolormesh(frequencies, amplitudes, mag, cmap="viridis")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [a.u.]")
    plt.colorbar(label="Signal [a.u.]")
    plt.show()
    return


@app.cell
def _(amplitudes, frequencies, mag, np, savgol_filter):
    # ------------------------------------------------------------------
    # 1) Row-normalize to remove amplitude-dependent background
    # ------------------------------------------------------------------
    baseline = np.median(mag, axis=1, keepdims=True)
    norm = mag / baseline                                # ~1 away from dips
    norm_s = savgol_filter(norm, window_length=11, polyorder=2, axis=1)

    # ------------------------------------------------------------------
    # 2) Detect dips per row, but only keep SIGNIFICANT ones
    #    Significance = depth relative to the row's MAD noise.
    # ------------------------------------------------------------------
    n_amp, n_freq = norm_s.shape
    row_median = np.median(norm_s, axis=1, keepdims=True)
    row_mad = np.median(np.abs(norm_s - row_median), axis=1, keepdims=True) + 1e-12
    z = (row_median - norm_s) / row_mad                  # positive => dip

    dip_idx = np.argmax(z, axis=1)                       # deepest dip per row
    dip_freq = frequencies[dip_idx]
    dip_z = z[np.arange(n_amp), dip_idx]

    SIGNIFICANCE = 5.0                                   # >5 sigma-ish
    valid = dip_z > SIGNIFICANCE

    print(f"Rows with a significant dip: {valid.sum()}/{n_amp}")

    # ------------------------------------------------------------------
    # 3) Cluster the valid dips into "low-power" vs "high-power" branches
    #    using a simple 1D split on frequency.
    # ------------------------------------------------------------------
    valid_freqs = dip_freq[valid]
    valid_amps  = amplitudes[valid]

    # Split by the midpoint between min and max valid dip frequency
    f_lo, f_hi = valid_freqs.min(), valid_freqs.max()
    split = 0.5 * (f_lo + f_hi)

    low_branch  = valid_freqs < split       # low-power dressed resonator (higher f here)
    high_branch = ~low_branch

    # In your data the DRESSED (low-power) resonator is at ~7.2015 GHz,
    # the BARE (high-power) resonator is at ~7.1993 GHz.
    # The dressed one is the branch that exists at LOW amplitudes.
    mean_amp_A = valid_amps[low_branch].mean()  if low_branch.any()  else np.inf
    mean_amp_B = valid_amps[high_branch].mean() if high_branch.any() else np.inf

    if mean_amp_A < mean_amp_B:
        dressed_mask, bare_mask = low_branch, high_branch
    else:
        dressed_mask, bare_mask = high_branch, low_branch

    f_dressed = np.median(valid_freqs[dressed_mask])
    f_bare    = np.median(valid_freqs[bare_mask])

    print(f"Dressed (low-power) f_r : {f_dressed/1e9:.6f} GHz")
    print(f"Bare    (high-power) f_r: {f_bare/1e9:.6f} GHz")
    print(f"Punchout shift          : {(f_dressed - f_bare)/1e6:+.3f} MHz")

    # ------------------------------------------------------------------
    # 4) Find punchout transition amplitude:
    #    highest amplitude at which the dressed branch is still visible.
    # ------------------------------------------------------------------
    dressed_amps = valid_amps[dressed_mask]
    punchout_amp = dressed_amps.max()
    lowest_dressed_amp = dressed_amps.min()

    # ------------------------------------------------------------------
    # 5) Choose readout point
    #    - freq  : dressed resonator (state-dependent response)
    #    - amp   : ~70% of punchout amplitude, i.e. as much power as we
    #              can safely use before leaving the dispersive regime.
    # ------------------------------------------------------------------
    SAFETY = 0.7
    best_amp = SAFETY * punchout_amp
    # clamp inside the swept range
    best_amp = float(np.clip(best_amp, amplitudes.min(), amplitudes.max()))
    best_freq = float(f_dressed)

    print("\n--- Recommended readout point ---")
    print(f"Readout frequency: {best_freq/1e9:.6f} GHz")
    print(f"Readout amplitude: {best_amp:.4f}  (punchout ≈ {punchout_amp:.4f})")

    return (
        bare_mask,
        best_amp,
        best_freq,
        dressed_mask,
        punchout_amp,
        valid_amps,
        valid_freqs,
        z,
    )


@app.cell
def _(
    amplitudes,
    bare_mask,
    best_amp,
    best_freq,
    dressed_mask,
    frequencies,
    mag,
    plt,
    punchout_amp,
    valid_amps,
    valid_freqs,
    z,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im = axes[0].pcolormesh(frequencies, amplitudes, mag, cmap="viridis", shading="auto")
    axes[0].scatter(valid_freqs[dressed_mask], valid_amps[dressed_mask],
                    s=12, c="red", label="dressed branch")
    axes[0].scatter(valid_freqs[bare_mask], valid_amps[bare_mask],
                    s=12, c="orange", label="bare branch")
    axes[0].axvline(best_freq, color="red", ls="--", lw=1)
    axes[0].axhline(best_amp,  color="red", ls="--", lw=1)
    axes[0].axhline(punchout_amp, color="orange", ls=":", lw=1, label="punchout amp")
    axes[0].set_xlabel("Frequency [Hz]"); axes[0].set_ylabel("Amplitude [a.u.]")
    axes[0].set_title("Raw magnitude")
    axes[0].legend(loc="lower right", fontsize=8)
    plt.colorbar(im, ax=axes[0])

    im2 = axes[1].pcolormesh(frequencies, amplitudes, z,
                             cmap="magma", shading="auto", vmin=0, vmax=10)
    axes[1].scatter(valid_freqs[dressed_mask], valid_amps[dressed_mask],
                    s=12, c="cyan")
    axes[1].scatter(valid_freqs[bare_mask], valid_amps[bare_mask],
                    s=12, c="orange")
    axes[1].axvline(best_freq, color="cyan", ls="--", lw=1)
    axes[1].axhline(best_amp,  color="cyan", ls="--", lw=1)
    axes[1].set_xlabel("Frequency [Hz]"); axes[1].set_ylabel("Amplitude [a.u.]")
    axes[1].set_title("Dip significance (z-score)")
    plt.colorbar(im2, ax=axes[1], label="σ")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(f0, np, ok, quality):
    print(f"n valid fits: {ok.sum()} / {len(ok)}")
    print(f"f0 range on valid rows: {np.nanmin(f0[ok]):.4e} .. {np.nanmax(f0[ok]):.4e}")
    print(f"quality percentiles: {np.percentile(quality[ok], [10, 50, 90])}")
    return


if __name__ == "__main__":
    app.run()
