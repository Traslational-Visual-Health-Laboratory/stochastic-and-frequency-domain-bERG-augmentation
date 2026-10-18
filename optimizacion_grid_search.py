"""
Hyperparameter optimization for the generative model (bERG) - parallel version.

Same methodology as the sequential version:
  - Per class (WITH EMD / WITHOUT EMD), 100 random real fragments form the
    reference set; the rest form the comparison set.
  - Real signals are read directly from the small csv files (VOLTAJE column).
  - For each comparison file, its source file (full signal) is looked up and
    model() is called on that source to generate the synthetic signal, once
    per each of the 3456 configurations.
  - The real-vs-real distribution does not depend on hyperparameters: it is
    computed once per class.
  - Per configuration, the synthetic-vs-reference distribution is computed,
    both are normalized (min-max), and the difference of means per metric is
    taken; the overall error is the sum over all metrics.

Differences from the sequential version:
  - The 3456 configurations of each class are distributed across several
    processes (ProcessPoolExecutor), taking advantage of the CPU cores.
  - Results are saved to the csv every 'save_every' completed configurations
    (it doesn't wait for everything to finish), so progress isn't lost if
    there's a power outage or the script is interrupted.
  - If the output csv already exists, the configurations already computed
    are read and skipped (resume). To start from scratch, delete the csv or
    set resume = False.

On Windows, multiprocessing requires that the code launching the processes
be protected by 'if __name__ == "__main__":'. This is a technical
requirement of the multiprocessing module (it uses 'spawn' and re-imports
the script in each child process), not a main() function to organize the
code.
"""

import os
import gc
import csv
import time
import random
import hashlib
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed (pip install tqdm). Continuing without a progress bar.")

    def tqdm(iterable, **kwargs):
        return iterable

from Stochastic_generator import model, calculate_metrics_and_entropies


# ============================================================
# general-purpose functions (used by the main process and the workers)
# ============================================================

def basename_no_ext(file_name):
    return os.path.splitext(file_name)[0]


def find_source_file(small_file_name, source_bases):
    """Finds which 'source' file a 'small' file corresponds to.

    Tries prefixes of the name (from longest to shortest, token by token)
    until it finds one that exists in 'source'.
    """
    base = basename_no_ext(small_file_name)
    tokens = base.split("_")

    for cut in range(len(tokens), 0, -1):
        candidate = "_".join(tokens[:cut])
        if candidate in source_bases:
            return source_bases[candidate]

    return None


def read_real_fragment(file_path):
    """Reads a small real fragment directly from the csv (VOLTAJE column)."""
    df = pd.read_csv(file_path)
    if "VOLTAJE" in df.columns:
        return df["VOLTAJE"].to_numpy(dtype=float)
    return df.iloc[:, 0].to_numpy(dtype=float)


def list_fragments(folder):
    files = sorted(
        entry.name
        for entry in os.scandir(folder)
        if entry.is_file() and entry.name.lower().endswith(".csv")
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No .csv files found in {folder}")
    return files


def split_reference_comparison(files, num_reference, rng):
    files = list(files)
    rng.shuffle(files)
    reference = files[:num_reference]
    comparison = files[num_reference:]
    return reference, comparison


def generate_synthetic(source_folder, source_file, fs, duration, num_states, alpha, beta,
                        window, order, max_retries=5):
    """Calls model() on the source file and returns only the synthetic signal."""
    retries = 0
    last_error = None
    while retries < max_retries:
        try:
            _, synthetic = model(
                source_folder,
                source_file,
                fs=fs,
                duration_sec=duration,
                num_states=num_states,
                alpha=alpha,
                beta=beta,
                window_length=window,
                poly_order=order,
            )
            return np.asarray(synthetic).ravel()
        except Exception as e:
            retries += 1
            last_error = e
    print(f"    could not generate synthetic for '{source_file}' after {max_retries} attempts: {last_error}")
    return None


def discover_metric_names(signal_a, signal_b, fs):
    m = calculate_metrics_and_entropies(signal_a, signal_b, fs=fs)
    return list(m.keys())


def pairwise_distribution(signals_a, signals_b, fs, metric_names,
                           num_samples_b=None, rng=None):
    """Computes the calculate_metrics_and_entropies metrics between pairs (a, b).

    If 'num_samples_b' is None (or >= len(signals_b)), every signal in 'a' is
    compared against ALL signals in 'b' (original behavior).

    If 'num_samples_b' is an integer smaller than len(signals_b), each signal
    in 'a' is compared only against a random sample of that size taken from
    'b' (a different sample for each 'sa'), instead of against all of them.
    This reduces the number of comparisons from len(a)*len(b) to
    len(a)*num_samples_b.

    'rng' should be a random.Random instance (so sampling is reproducible);
    if not passed, the global 'random' module is used.
    """
    values = {k: [] for k in metric_names}
    generator = rng if rng is not None else random
    use_sampling = num_samples_b is not None and num_samples_b < len(signals_b)

    for sa in signals_a:
        if use_sampling:
            indices = generator.sample(range(len(signals_b)), num_samples_b)
            signals_b_used = [signals_b[i] for i in indices]
        else:
            signals_b_used = signals_b

        for sb in signals_b_used:
            try:
                m = calculate_metrics_and_entropies(sa, sb, fs=fs)
            except Exception:
                continue
            for k in metric_names:
                values[k].append(m[k])
    return {k: np.array(v, dtype=float) for k, v in values.items()}


def normalized_mean_difference(real_values, synthetic_values):
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return np.nan
    combined = np.concatenate([real_values, synthetic_values])
    vmin, vmax = np.min(combined), np.max(combined)
    if (vmax - vmin) < 1e-15:
        return 0.0
    real_norm = (real_values - vmin) / (vmax - vmin)
    synth_norm = (synthetic_values - vmin) / (vmax - vmin)
    return abs(np.mean(real_norm) - np.mean(synth_norm))


def write_csv_rows(csv_path, rows, fields, write_header):
    if len(rows) == 0:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def seed_for_config(config, base_seed):
    """Generates a reproducible integer seed from a configuration and a base
    seed. hash() is not used because for strings it is not reproducible
    across different python runs (random PYTHONHASHSEED); with md5 over the
    text, the same number is always obtained.
    """
    text = f"{base_seed}_{config}"
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)


def read_completed_configs(csv_path):
    """Reads the output csv (if it exists) and returns the set of
    configurations already computed, as tuples
    (class, num_states, alpha, beta, sg_window, sg_poly).
    """
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        key = (
            row["clase"],
            int(row["num_states"]),
            float(row["alpha"]),
            float(row["beta"]),
            int(row["sg_window"]),
            int(row["sg_poly"]),
        )
        completed.add(key)
    return completed


# ============================================================
# global state of each worker process (filled in by init_worker)
# ============================================================

g_source_folder = None
g_comparison_sources = None
g_reference_signals = None
g_real_distribution = None
g_fs = None
g_duration = None
g_metric_names = None
g_num_reference_samples = None
g_base_seed = None


def init_worker(source_folder, comparison_sources, reference_signals,
                 real_distribution, fs, duration, metric_names,
                 num_reference_samples, base_seed):
    global g_source_folder, g_comparison_sources, g_reference_signals
    global g_real_distribution, g_fs, g_duration, g_metric_names
    global g_num_reference_samples, g_base_seed
    g_source_folder = source_folder
    g_comparison_sources = comparison_sources
    g_reference_signals = reference_signals
    g_real_distribution = real_distribution
    g_fs = fs
    g_duration = duration
    g_metric_names = metric_names
    g_num_reference_samples = num_reference_samples
    g_base_seed = base_seed


def process_config(config):
    """Runs in a worker process. config = (class_id, num_states, alpha, beta, window, order)."""
    class_id, num_states, alpha, beta, window, order = config

    comparison_synthetics = []
    for source_file in g_comparison_sources:
        if source_file is None:
            continue
        synthetic = generate_synthetic(
            g_source_folder, source_file, g_fs, g_duration, num_states, alpha, beta, window, order
        )
        if synthetic is not None:
            comparison_synthetics.append(synthetic)

    if len(comparison_synthetics) == 0:
        return None

    # reproducible rng per configuration: every run with the same base seed
    # and the same configuration will generate the same reference sample,
    # even if the worker that processes it changes between runs.
    config_rng = random.Random(seed_for_config(config, g_base_seed))

    synthetic_distribution = pairwise_distribution(
        comparison_synthetics, g_reference_signals, g_fs, g_metric_names,
        num_samples_b=g_num_reference_samples, rng=config_rng,
    )

    row = {
        "clase": class_id,
        "num_states": num_states,
        "alpha": alpha,
        "beta": beta,
        "sg_window": window,
        "sg_poly": order,
    }
    overall_error = 0.0
    for metric in g_metric_names:
        d = normalized_mean_difference(
            g_real_distribution.get(metric, np.array([])),
            synthetic_distribution.get(metric, np.array([])),
        )
        row[f"error_{metric}"] = d
        if not np.isnan(d):
            overall_error += d
    row["error_global"] = overall_error

    # helps the worker process free any intermediate objects for this
    # configuration (synthetic signals, distributions, etc.) as soon as
    # possible, instead of waiting for the garbage collector to decide on
    # its own.
    del comparison_synthetics, synthetic_distribution
    gc.collect()

    return row


# ============================================================
# per-class orchestration (runs in the main process)
# ============================================================

def process_class(class_id, fragments_folder, source_folder, num_reference, fs, duration,
                   states, alpha, beta, sg_window, sg_order, rng, output_file,
                   num_processes, save_every, completed_configs,
                   num_reference_samples, base_seed):

    print(f"\n=== Processing class: {class_id} ===")

    fragment_files = list_fragments(fragments_folder)
    if len(fragment_files) <= num_reference:
        raise ValueError(
            f"'{fragments_folder}' has {len(fragment_files)} fragments, "
            f"more than {num_reference} are needed to split reference and comparison."
        )

    reference_files, comparison_files = split_reference_comparison(
        fragment_files, num_reference, rng
    )
    print(f"  total fragments: {len(fragment_files)} | reference: {len(reference_files)} | comparison: {len(comparison_files)}")

    real_reference_signals = [
        read_real_fragment(os.path.join(fragments_folder, f)) for f in reference_files
    ]
    real_comparison_signals = [
        read_real_fragment(os.path.join(fragments_folder, f)) for f in comparison_files
    ]

    metric_names = discover_metric_names(
        real_comparison_signals[0], real_reference_signals[0], fs
    )

    # this distribution doesn't depend on any hyperparameter: it's computed once.
    # it uses the same reference-sampling logic as the configurations, so the
    # comparison (real-vs-real vs synthetic-vs-real) is consistent.
    real_rng = random.Random(seed_for_config((class_id, "real_vs_real"), base_seed))
    real_distribution = pairwise_distribution(
        real_comparison_signals, real_reference_signals, fs, metric_names,
        num_samples_b=num_reference_samples, rng=real_rng,
    )

    source_files = [
        entry.name
        for entry in os.scandir(source_folder)
        if entry.is_file() and entry.name.lower().endswith(".csv")
    ]
    source_bases = {basename_no_ext(f): f for f in source_files}

    comparison_sources = []
    for small_file_name in comparison_files:
        source_file = find_source_file(small_file_name, source_bases)
        if source_file is None:
            print(f"  warning: no source file found for '{small_file_name}', it is omitted from the comparison set")
        comparison_sources.append(source_file)

    combinations = [
        (n, a, b, v, o)
        for (n, a, b, v, o) in itertools.product(states, alpha, beta, sg_window, sg_order)
        if v > o  # savgol_filter requires window_length > polyorder; otherwise model() fails
    ]
    invalid_combinations = (
        len(states) * len(alpha) * len(beta) * len(sg_window) * len(sg_order)
        - len(combinations)
    )
    if invalid_combinations > 0:
        print(f"  combinations discarded due to window <= order (invalid SG): {invalid_combinations}")

    pending = [
        (class_id, n, a, b, v, o) for (n, a, b, v, o) in combinations
        if (class_id, n, a, b, v, o) not in completed_configs
    ]
    already_done = len(combinations) - len(pending)
    print(f"  total configurations: {len(combinations)} | already computed: {already_done} | pending: {len(pending)}")

    if len(pending) == 0:
        return

    fields = None
    write_header = not os.path.exists(output_file)

    # processed in batches of size 'save_every': once a batch finishes, the
    # process pool is fully shut down (freeing all its memory) and the csv is
    # saved, before opening a new pool for the next batch.
    batches = [pending[i:i + save_every] for i in range(0, len(pending), save_every)]

    bar = tqdm(total=len(pending), desc=f"{class_id}")
    for batch in batches:
        row_buffer = []

        with ProcessPoolExecutor(
            max_workers=num_processes,
            initializer=init_worker,
            initargs=(source_folder, comparison_sources, real_reference_signals,
                      real_distribution, fs, duration, metric_names,
                      num_reference_samples, base_seed),
        ) as pool:

            futures = [pool.submit(process_config, config) for config in batch]

            for future in as_completed(futures):
                row = future.result()
                bar.update(1)
                if row is None:
                    continue
                if fields is None:
                    fields = list(row.keys())
                row_buffer.append(row)

        # the 'with' block has ended: this batch's processes were shut down
        # and their memory freed before saving and moving to the next batch
        if len(row_buffer) > 0:
            write_csv_rows(output_file, row_buffer, fields, write_header)
            write_header = False

    bar.close()


# ============================================================
# configuration and execution
# ============================================================

if __name__ == "__main__":

    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    fs = 200
    duration_sec = 20
    num_reference = 100

    num_reference_samples = 100

    fragments_folder_with_emd = r""
    fragments_folder_without_emd = r""

    # folders with the full signals that model() needs, analogous to
    # 'source' in generate_lote. adjust these paths to your actual folders.
    source_folder_with_emd = r"/"
    source_folder_without_emd = r"/"

    states = [100, 200, 300, 400, 500, 600]
    alpha = [1e-20, 1e-10, 1e-5, 1e-3]
    beta = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sg_window = [5, 15, 25, 35, 45, 55]
    sg_order = [2, 3, 4, 5]

    output_file = "1.csv"

    # leave this as True to be able to resume after a power outage; set it to
    # False (or delete the csv) if you want to start everything from scratch.
    resume = True
    if not resume and os.path.exists(output_file):
        os.remove(output_file)

    # number of parallel processes. leaves headroom for the rest of the system.
    num_processes = max(1, os.cpu_count() - 2)

    # how often (in completed configurations) results are written to the csv
    save_every = 20

    completed_configs = read_completed_configs(output_file)

    t0 = time.time()

    process_class(
        "con_emd", fragments_folder_with_emd, source_folder_with_emd, num_reference, fs,
        duration_sec, states, alpha, beta, sg_window, sg_order, rng, output_file,
        num_processes, save_every, completed_configs,
        num_reference_samples, seed,
    )

    process_class(
        "sin_emd", fragments_folder_without_emd, source_folder_without_emd, num_reference, fs,
        duration_sec, states, alpha, beta, sg_window, sg_order, rng, output_file,
        num_processes, save_every, completed_configs,
        num_reference_samples, seed,
    )

    print(f"\nDone. results saved to: {output_file}")
    print(f"total time: {(time.time() - t0) / 60:.1f} minutes")