import csv
import numpy as np

def load_debugging_csv(csv_path):
    """
    Load a debugging CSV into a nested dictionary keyed by field name and wavelength index.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file written by save_debugging_iteration.

    Returns
    -------
    dict
        Nested dict where dict[fieldname][lam_index] is a np.ndarray of values
        across all iterations. 'iteration' and 'lam_index' are also included.
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = reader.fieldnames
    lam_indices = sorted(set(int(r['lam_index']) for r in rows))

    result = {}
    for field in fieldnames:
        result[field] = {}
        for j in lam_indices:
            values = [float(r[field]) for r in rows if int(r['lam_index']) == j]
            result[field][j] = np.array(values)

    return result

def load_contrast_csv(csv_path):
    """
    Load a measured contrast CSV written by save_outputs.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (e.g. 'measured_contrast.csv' or 'predicted_contrast.csv').

    Returns
    -------
    contrast : np.ndarray, shape (niter,)
        Contrast values, one per iteration.
    """
    return np.loadtxt(csv_path, delimiter=',', skiprows=1)