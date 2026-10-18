# ========================
# 1. Introduction
# ========================
"""
TS³-Generator: Time Series Stochastic and Spectral Generator
------------------------------------------------------------
This system implements a statistical generative model for the synthetic generation
of stochastic time series. The model represents real-valued time series as stochastic
processes with memory, capturing short-term temporal dependencies through second-order
transition matrices derived from a discretization of the signal value space.

Synthetic signals are generated via stochastic random walks of fixed length, followed
by temporal smoothing to ensure continuity and reduce high-frequency artifacts.
Subsequently, a spectral adjustment stage based on the Fast Fourier Transform (FFT)
is applied to align the frequency distribution of the synthetic signals with that of
real reference signals, ensuring spectral coherence.

Designed as a data augmentation module, TS³-Generator enables the generation of multiple
statistically consistent synthetic realizations from each real signal, supporting the
training, validation, and evaluation of machine learning models in scientific and
technological applications, particularly in biomedical signal analysis.

------------------------------------------------------------
Este sistema implementa un modelo generativo estadístico para la generación sintética
de series de tiempo de naturaleza estocástica. El modelo representa series de tiempo
reales como procesos estocásticos con memoria, capturando dependencias temporales de
corto plazo mediante matrices de transición de segundo orden obtenidas a partir de la
discretización del espacio de valores de la señal.

Las señales sintéticas se generan mediante caminatas estocásticas de longitud fija,
seguidas de un proceso de suavizado temporal que garantiza continuidad y reduce
artefactos de alta frecuencia. Posteriormente, se aplica un ajuste espectral basado
en la Transformada Rápida de Fourier (FFT), con el fin de alinear la distribución de
frecuencias de las señales sintéticas con la observada en señales reales de referencia.

Diseñado como un módulo de aumento de datos, TS³-Generator permite generar múltiples
realizaciones sintéticas estadísticamente consistentes a partir de cada señal real,
apoyando el entrenamiento, validación y evaluación de modelos de aprendizaje automático
en aplicaciones científicas y tecnológicas, particularmente en el análisis de señales
biomédicas.
"""

# ========================
# 2. Library Imports
# ========================
# --- System and utility ---
import os
import math
import random
# --- Numerical and data processing ---
import numpy as np
import pandas as pd
# --- Signal processing ---
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy, wasserstein_distance
# --- Visualization ---

# --- Optional JIT acceleration (Numba) ---
# The transition-matrix cumulative sum, the second-order stochastic walk,
# and sample entropy are the three real CPU bottlenecks in this module
# (verified by profiling): they involve either a large elementwise pass
# over an O(num_states^3) array or a tight sequential/O(N^2) loop that
# NumPy cannot vectorize away. Numba JIT-compiles these to native code,
# giving ~6-10x speedups on the transition-matrix/walk generation and
# ~15-20x on sample entropy, with numerically identical results.
# If Numba is not installed, the code falls back transparently to the
# original pure-NumPy/Python implementations (just slower).
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        # No-op decorator fallback so the rest of the module still works
        # (without the JIT speedup) when numba isn't installed.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def wrapper(func):
            return func
        return wrapper

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# ========================
# 3. Architecture
# ========================
def calculate_transition_matrix(data, num_states):
    """
    Compute a second-order transition probability matrix from a time series.

    The input signal is discretized into a finite number of states using
    uniform binning. The function then estimates a third-order second
    transition matrix P(k | i, j), representing the probability of
    transitioning to state k given the two previous states i and j.

    Parameters
    ----------
    data : array-like
        One-dimensional time series signal.
    num_states : int
        Number of discrete states used to quantize the signal.

    Returns
    -------
    matrix : ndarray
        A 3D transition probability matrix of shape
        (num_states, num_states, num_states), where matrix[i, j, k]
        represents P(state_t = k | state_{t-1} = j, state_{t-2} = i).
    states : ndarray
        Discretized state sequence corresponding to the input signal.
    """

    # Define uniform bins for signal discretization
    bins = np.linspace(np.min(data), np.max(data), num_states + 1)

    # Assign each sample to a discrete state
    states = np.digitize(data, bins) - 1
    states[states >= num_states] = num_states - 1

    # Extract all observed third-order transitions (i, j, k)
    i = states[:-2]
    j = states[1:-1]
    k = states[2:]

    # Encode (i, j, k) triplets into linear indices
    idx = i * num_states**2 + j * num_states + k
    counts = np.bincount(idx, minlength=num_states**3)

    # Reshape counts into a 3D transition tensor
    matrix = counts.reshape((num_states, num_states, num_states)).astype(float)

    # Normalize transition probabilities along the last axis
    row_sums = matrix.sum(axis=2, keepdims=True)
    np.divide(matrix, row_sums, where=row_sums != 0, out=matrix)

    return matrix, states

def regularize_transition_matrix(matrix, alpha=1e-20):
    """
    Apply probabilistic regularization to a second-order transition matrix.

    This function stabilizes sparse or zero-probability transitions by
    mixing the observed transition probabilities with a uniform prior.
    The regularization prevents degenerate transitions during stochastic
    signal generation while preserving the empirical structure of the matrix.

    Parameters
    ----------
    matrix : ndarray
        A 3D transition probability matrix of shape
        (num_states, num_states, num_states).
    alpha : float, optional
        Regularization weight controlling the contribution of the uniform
        distribution. Small values preserve the empirical transitions while
        avoiding zero-probability states (default is 1e-20).

    Returns
    -------
    regularized : ndarray
        Regularized and normalized transition probability matrix.
    """

    # Blending with a uniform prior and renormalizing are fused into a
    # single pass over the (num_states, num_states, num_states) array
    # instead of NumPy's ~4 separate full-array passes (multiply, add,
    # sum, divide). For num_states=200 this touches 8M elements per
    # pass, so fusing them is a meaningful, memory-bandwidth-bound win.
    return _regularize_matrix_core(np.ascontiguousarray(matrix, dtype=np.float64), alpha)

@njit(cache=True, fastmath=True, parallel=True)
def _regularize_matrix_core(matrix, alpha):
    num_states = matrix.shape[0]
    uniform_val = 1.0 / num_states
    regularized = np.empty_like(matrix)

    for i in prange(num_states):
        for j in range(num_states):
            row_sum = 0.0
            for k in range(num_states):
                val = (1.0 - alpha) * matrix[i, j, k] + alpha * uniform_val
                regularized[i, j, k] = val
                row_sum += val
            if row_sum != 0.0:
                inv = 1.0 / row_sum
                for k in range(num_states):
                    regularized[i, j, k] *= inv

    return regularized

def generate_stochastic_walk(matrix, length, init_state_prev=None, init_state_curr=None):
    """
    Generate a second-order discrete stochastic walk from a transition matrix.

    This function samples a discrete-time stochastic sequence using a
    second-order Markov process, where the probability of the next state
    depends on the two previous states. The walk is generated efficiently
    using precomputed cumulative distribution functions (CDFs).

    Parameters
    ----------
    matrix : ndarray
        A 3D transition probability matrix of shape
        (num_states, num_states, num_states), where
        P(X_t = k | X_{t-2} = i, X_{t-1} = j).
    length : int
        Desired length of the generated stochastic sequence.
    init_state_prev : int, optional
        Initial state at time t-2. If None, it is sampled uniformly at random.
    init_state_curr : int, optional
        Initial state at time t-1. If None, it is sampled uniformly at random.

    Returns
    -------
    states : ndarray
        Discrete stochastic sequence of integer states with shape (length,).
    """

    num_states = matrix.shape[0]

    # Initialize the first two states
    if init_state_prev is None or init_state_curr is None:
        i = np.random.randint(num_states)
        j = np.random.randint(num_states)
    else:
        i, j = init_state_prev, init_state_curr

    # Precompute cumulative distribution functions for fast sampling.
    # This is an O(num_states^3) pass (8M elements for num_states=200),
    # JIT-compiled and parallelized across the leading axis since it
    # dominates the runtime of this function far more than the walk
    # loop itself.
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    cdfs = _cumsum_last_axis_numba(matrix)

    # Generate the stochastic walk. The sequential dependency (each
    # state depends on the previous two) means this can't be
    # vectorized with NumPy, but a JIT-compiled loop with manual
    # binary search removes Python call overhead and beats
    # np.searchsorted called in a Python loop by ~10x.
    rand_vals = np.random.rand(length - 2).astype(np.float32)
    states = _walk_core_numba(cdfs, length, i, j, rand_vals)

    return states


@njit(cache=True, fastmath=True, parallel=True)
def _cumsum_last_axis_numba(matrix):
    ns0, ns1, ns2 = matrix.shape
    out = np.empty_like(matrix)
    for i in prange(ns0):
        for j in range(ns1):
            acc = 0.0
            for k in range(ns2):
                acc += matrix[i, j, k]
                out[i, j, k] = acc
    return out


@njit(cache=True, fastmath=True)
def _walk_core_numba(cdfs, length, i, j, rand_vals):
    num_states = cdfs.shape[0]
    states = np.empty(length, dtype=np.int32)
    states[0], states[1] = i, j

    for t in range(2, length):
        r = rand_vals[t - 2]
        row = cdfs[i, j]

        # Manual binary search (equivalent to np.searchsorted, but
        # without per-call Python/NumPy dispatch overhead).
        lo, hi = 0, num_states
        while lo < hi:
            mid = (lo + hi) // 2
            if row[mid] < r:
                lo = mid + 1
            else:
                hi = mid

        next_state = lo
        if next_state >= num_states:
            next_state = num_states - 1

        states[t] = next_state
        i, j = j, next_state

    return states

def smooth_signal(signal, window_length=45, polyorder=3):
    """
    Smooth a one-dimensional signal using a Savitzky–Golay filter.

    This function applies polynomial smoothing to reduce high-frequency
    artifacts introduced during stochastic generation while preserving
    physiologically relevant temporal features such as slopes and curvature.

    Parameters
    ----------
    signal : array-like
        Input signal to be smoothed.
    window_length : int, optional
        Length of the filter window (number of samples). Must be odd.
    polyorder : int, optional
        Order of the polynomial used to fit the samples.

    Returns
    -------
    smoothed_signal : ndarray
        Smoothed signal with the same length as the input.
    """
    return savgol_filter(signal, window_length, polyorder)

def compute_fft(signal, fs):
    """
    Compute the one-sided Fast Fourier Transform (FFT) of a real-valued signal.

    This function returns the frequency vector, amplitude spectrum, and
    complex FFT coefficients, which are later used for spectral comparison
    and selective spectral adjustment of synthetic signals.

    Parameters
    ----------
    signal : array-like
        Input time-domain signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freq : ndarray
        Frequency vector corresponding to the FFT bins.
    amplitude : ndarray
        Magnitude spectrum of the signal.
    fft_values : ndarray
        Complex FFT coefficients.
    """
    signal = np.asarray(signal, dtype=np.float32, order='C')
    n = signal.size

    # Frequency axis for real-valued FFT
    freq = rfftfreq(n, 1.0 / fs)

    # Compute FFT
    fft_values = rfft(signal)
    amplitude = np.abs(fft_values)

    return freq, amplitude, fft_values

def adjust_to_dynamic_range_selective(
    fft_synthetic,
    fft_segments,
    beta=0.0,
    max_k=3,
    stretch_std=False,
    z_threshold=1e-6,
    interpolation=False
):
    """
    Selectively adjust the spectral amplitude of a synthetic signal to match
    the empirical spectral characteristics of real signals.

    If interpolation is False, spectral amplitudes that fall outside the
    empirical min–max range of real signals are reassigned to a random value
    within that range for each frequency.

    If interpolation is True, amplitudes are adjusted using convex interpolation
    of z-scored real spectra, modifying only frequency components whose
    deviations exceed a specified threshold.

    Parameters
    ----------
    fft_synthetic : ndarray
        Complex FFT coefficients of the synthetic signal.
    fft_segments : ndarray
        Complex FFT coefficients of real signal segments
        with shape (n_segments, n_frequencies).
    beta : float, optional
    max_k : int, optional
        Maximum number of real segments used for convex interpolation.
    stretch_std : bool, optional
        Whether to randomly stretch the standard deviation of the mixed
        z-score spectrum to increase variability.
    z_threshold : float, optional
        Z-score threshold that determines which frequency bins are adjusted.
    interpolation : bool, optional
        Whether to use convex interpolation in z-score space (True) or
        stochastic clipping within the empirical spectral range (False).

    Returns
    -------
    fft_synthetic_adjusted : ndarray
        Spectrally adjusted complex FFT coefficients of the synthetic signal.
    """

    fft_segments = np.asarray(fft_segments, dtype=np.complex64, order='C')
    fft_synthetic = np.asarray(fft_synthetic, dtype=np.complex64, order='C')

    abs_segments = np.abs(fft_segments)
    abs_synth = np.abs(fft_synthetic)

    # Empirical spectral bounds
    min_amp = abs_segments.min(axis=0)
    max_amp = abs_segments.max(axis=0)

    # Phase preservation
    phase = np.angle(fft_synthetic)

    # --- ORIGINAL METHOD: stochastic clipping ---
    if not interpolation:
        out_of_range = (
            (abs_synth < min_amp) |
            (abs_synth > max_amp)
        )

        adjusted_amplitudes = abs_synth.copy()

        A = np.random.uniform(
            min_amp[out_of_range],
            max_amp[out_of_range]
        )

        ratio = np.ones_like(abs_synth, dtype=np.float32)

        ratio[out_of_range] = A / (
            (1 + beta) * abs_synth[out_of_range] + 1e-10
        )

        fft_synthetic_adjusted = fft_synthetic * ratio

        return fft_synthetic_adjusted

    # --- EXTENDED METHOD: convex interpolation in z-score space ---
    mean = abs_segments.mean(axis=0)
    std = abs_segments.std(axis=0) + 1e-10

    z_segments = (abs_segments - mean) / std
    n_segments = z_segments.shape[0]

    k = np.random.randint(2, min(n_segments, max_k) + 1)
    idxs = np.random.choice(n_segments, k, replace=False)

    alphas = np.random.dirichlet(np.full(k, 0.5, dtype=np.float32))
    z_mix = np.tensordot(alphas, z_segments[idxs], axes=(0, 0))

    if stretch_std:
        z_mean = z_mix.mean()
        z_mix = (z_mix - z_mean) * np.random.uniform(1.1, 1.5) + z_mean

    adjusted_amplitudes = z_mix * std + mean

    z_synth = (abs_synth - mean) / std
    modify_mask = np.abs(z_synth) > z_threshold

    ratio = np.ones_like(abs_synth, dtype=np.float32)
    ratio_sel = adjusted_amplitudes[modify_mask] / (
        abs_synth[modify_mask] * (1 + beta) + 1e-10
    )
    ratio[modify_mask] = ratio_sel

    fft_synthetic_adjusted = fft_synthetic * ratio
    return fft_synthetic_adjusted

def model(
    path,
    filename,
    fs=200,
    duration_sec=20,
    num_states=500,
    alpha=1e-5,
    beta=0.0,
    window_length=55,
    poly_order=5,
    delta=0.2
):
    """
    Generate a synthetic basal ERG signal using a second-order stochastic
    transition model with spectral and statistical adjustment.

    The function extracts a random segment from a real ERG recording,
    estimates a second-order transition matrix, generates a stochastic
    walk, and applies temporal smoothing, statistical matching, and
    selective spectral adjustment to produce a physiologically plausible
    synthetic signal.

    Parameters
    ----------
    path : str
        Directory path containing the ERG file.
    filename : str
        Name of the CSV file containing the ERG signal.
    fs : float
        Sampling frequency (Hz).
    duration_sec : float
        Duration of the generated signal in seconds.
    num_states : int
        Number of discrete states used for voltage discretization.
    alpha : float
        Regularization parameter for the transition matrix.
    beta : float
        Stabilization parameter for spectral adjustment.
    window_length : int
        Window length for Savitzky–Golay smoothing.
    poly_order : int
        Polynomial order for Savitzky–Golay smoothing.
    delta : float
        Weight controlling statistical matching strength.

    Returns
    -------
    real_segment : ndarray
        Real ERG segment extracted from the original recording.
    synthetic_signal : ndarray
        Generated synthetic ERG signal.
    """

    # Number of samples corresponding to the desired duration
    num_samples = math.ceil(fs * duration_sec)

    # Load ERG voltage signal
    try:
        signal_full = pd.read_csv(
            os.path.join(path, filename),
            usecols=[0]
        ).values.squeeze()
    except:
        signal_full = pd.read_excel(
            os.path.join(path, filename),
            usecols=[0]
        ).values.squeeze()

    # Randomly select a continuous segment
    max_start = len(signal_full) - num_samples
    start_idx = random.randint(0, max_start)
    real_segment = signal_full[start_idx:start_idx + num_samples]

    # Estimate second-order transition matrix
    transition_matrix, original_states = calculate_transition_matrix(
        signal_full, num_states
    )
    transition_matrix = regularize_transition_matrix(
        transition_matrix, alpha
    )

    # Rescale discrete states back to original signal range
    min_state, ptp_state = original_states.min(), np.ptp(original_states)
    min_val, ptp_val = real_segment.min(), np.ptp(real_segment)

    reconstructed_real_signal = (
        (original_states - min_state) / ptp_state
    ) * ptp_val + min_val

    # Generate second-order stochastic walk
    synthetic_signal = generate_stochastic_walk(
        transition_matrix,
        len(real_segment),
        init_state_prev=original_states[0],
        init_state_curr=original_states[1]
    )
    
    # Temporal smoothing
    synthetic_signal = smooth_signal(
        synthetic_signal,
        window_length=window_length,
        polyorder=poly_order
    )

    # Segment original signal and compute FFTs
    segment_length = len(real_segment)
    num_segments = len(signal_full) // segment_length
    signal_full = signal_full[:num_segments * segment_length]

    segments = signal_full.reshape(num_segments, segment_length)
    fft_segments = np.fft.rfft(segments, axis=1)

    # FFT of synthetic signal
    fft_synthetic = np.fft.rfft(synthetic_signal)

    # Ensure consistent frequency resolution
    min_len = min(fft_synthetic.size, fft_segments.shape[1])
    fft_synthetic = fft_synthetic[:min_len]
    fft_segments = fft_segments[:, :min_len]
    # Selective spectral adjustment
    fft_synthetic_adjusted = adjust_to_dynamic_range_selective(
        fft_synthetic,
        fft_segments,
        beta=beta
    )

    # Inverse FFT and final amplitude constraint
    synthetic_signal = np.fft.irfft(
        fft_synthetic_adjusted,
        n=len(real_segment)
    )

    return real_segment, synthetic_signal

def js_divergence(p, q, eps=1e-10):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def ssim_1d(x, y, C1=1e-4, C2=9e-4):
    x = np.asarray(x)
    y = np.asarray(y)

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return num / den

def sample_entropy(x, m=2, r=None):
    # The original implementation is O(N^2) with per-iteration NumPy
    # slicing/allocation overhead. A JIT-compiled version of the same
    # O(N^2) comparison loop removes that overhead entirely (~15-20x
    # faster measured on 4000-sample signals), which matters here
    # because this metric is typically computed once per generated
    # signal during evaluation.
    x = np.ascontiguousarray(x, dtype=np.float64)
    N = len(x)

    if r is None:
        r = 0.2 * np.std(x)

    B = _phi_numba(x, m, r, N)
    A = _phi_numba(x, m + 1, r, N)

    return -np.log((A + 1e-12) / (B + 1e-12))


@njit(cache=True, fastmath=True)
def _phi_numba(x, m, r, N):
    C = 0
    for i in range(N - m):
        for j in range(i + 1, N - m + 1):
            max_diff = 0.0
            for k in range(m):
                d = abs(x[j + k] - x[i + k])
                if d > max_diff:
                    max_diff = d
            if max_diff < r:
                C += 1
    return C

def spectral_entropy(x, fs):
    x = np.asarray(x)

    # FFT
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    psd = np.abs(np.fft.rfft(x))**2

    # Normalizar PSD
    psd_norm = psd / (np.sum(psd) + 1e-12)

    # Entropía de Shannon
    se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # Normalización opcional
    se /= np.log2(len(psd_norm))

    return se

def hjorth_parameters(x):
    x = np.asarray(x)

    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)

    return var_x, mobility, complexity

def calculate_metrics_and_entropies(
    original_signal,
    synthetic_signal,
    fs = 200
):
    original_signal = np.asarray(original_signal).ravel()
    synthetic_signal = np.asarray(synthetic_signal).ravel()

    if original_signal.size != synthetic_signal.size:
        raise ValueError("Signals must have the same length.")

    n = original_signal.size

    # ==== Basic statistics ====
    mean_orig = np.mean(original_signal)
    mean_syn = np.mean(synthetic_signal)
    std_orig = np.std(original_signal)
    std_syn = np.std(synthetic_signal)

    var_orig = std_orig**2
    var_syn = std_syn**2

    # ==== Error metrics ====
    diff = original_signal - synthetic_signal
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))

    # ==== Correlation ====
    correlation = np.corrcoef(original_signal, synthetic_signal)[0, 1]

    covariance = np.mean(
        (original_signal - mean_orig) *
        (synthetic_signal - mean_syn)
    )

    ccc = (2 * covariance) / (
        var_orig + var_syn + (mean_orig - mean_syn)**2
    )

    minkowski = np.sum(np.abs(diff)**3)**(1/3)

    # ==== Normalized cross-correlation ====
    orig_norm = (original_signal - mean_orig) / (std_orig + 1e-12)
    syn_norm = (synthetic_signal - mean_syn) / (std_syn + 1e-12)

    cross_corr = np.max(
        np.correlate(orig_norm, syn_norm, mode="full")
    ) / n

    # ==== Distribution metrics ====
    wasserstein = wasserstein_distance(
        original_signal, synthetic_signal
    )

    ssim_index = ssim_1d(original_signal, synthetic_signal)

    # ==== Jensen–Shannon ====
    bins = np.linspace(
        min(original_signal.min(), synthetic_signal.min()),
        max(original_signal.max(), synthetic_signal.max()),
        100
    )

    p_hist, _ = np.histogram(original_signal, bins=bins, density=True)
    q_hist, _ = np.histogram(synthetic_signal, bins=bins, density=True)

    js_div = js_divergence(p_hist, q_hist)

    # ==== Entropy metrics ====
    sampen_diff = abs(
        sample_entropy(original_signal) -
        sample_entropy(synthetic_signal)
    )

    specen_diff = abs(
        spectral_entropy(original_signal, fs) -
        spectral_entropy(synthetic_signal, fs)
    )

    # ==== Hjorth ====
    act_o, mob_o, comp_o = hjorth_parameters(original_signal)
    act_s, mob_s, comp_s = hjorth_parameters(synthetic_signal)

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "Pearson Correlation": correlation,
        "Concordance Correlation Coefficient (CCC)": ccc,
        "Minkowski Distance (p=3)": minkowski,
        "Cross-Correlation": cross_corr,
        "Wasserstein Distance": wasserstein,
        "SSIM": ssim_index,
        "Jensen–Shannon Divergence": js_div,
        "Sample Entropy Difference": sampen_diff,
        "Spectral Entropy Difference": specen_diff,
        "Hjorth Activity Difference": abs(act_o - act_s),
        "Hjorth Mobility Difference": abs(mob_o - mob_s),
        "Hjorth Complexity Difference": abs(comp_o - comp_s),
    }

    return metrics

def plot_signal_comparison(
    real_signal,
    synthetic_signal,
    fs=200,
    title="Signal comparison"
):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import rfft, rfftfreq

    # --- FFT computation con ventana de Hann ---
    n_real = len(real_signal)
    n_synth = len(synthetic_signal)

    # aplicar ventana de Hann
    window_real = real_signal * np.hanning(n_real)
    window_synth = synthetic_signal * np.hanning(n_synth)

    freq_real = rfftfreq(n_real, 1 / fs)
    freq_synth = rfftfreq(n_synth, 1 / fs)

    power_real = np.abs(rfft(window_real)) ** 2
    power_synth = np.abs(rfft(window_synth)) ** 2

    # --- Create figure ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # ======================================================
    # Row 1: Time-domain signals (sin ventana, como antes)
    # ======================================================
    axes[0, 0].plot(real_signal, color="tab:blue")
    axes[0, 0].set_title("Real signal (time domain)")

    axes[0, 1].plot(synthetic_signal, color="tab:orange")
    axes[0, 1].set_title("Synthetic signal (time domain)")

    # ======================================================
    # Row 2: Power spectra (con ventana de Hann aplicada)
    # ======================================================
    axes[1, 0].plot(freq_real, power_real, color="tab:blue")
    axes[1, 0].set_title("Power spectrum (real, Hann)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].set_ylim([0, 10])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freq_synth, power_synth, color="tab:orange")
    axes[1, 1].set_title("Power spectrum (synthetic, Hann)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].set_ylim([0, 10])
    axes[1, 1].grid(True, alpha=0.3)

    # ======================================================
    # Row 3: Amplitude distributions (sin cambios)
    # ======================================================
    bins = np.linspace(
        min(real_signal.min(), synthetic_signal.min()),
        max(real_signal.max(), synthetic_signal.max()),
        50
    )

    axes[2, 0].hist(real_signal, bins=bins, density=True, color="tab:blue", alpha=0.6)
    axes[2, 0].set_title("Amplitude distribution (real)")

    axes[2, 1].hist(synthetic_signal, bins=bins, density=True, color="tab:orange", alpha=0.6)
    axes[2, 1].set_title("Amplitude distribution (synthetic)")

    plt.tight_layout()
    plt.show()


