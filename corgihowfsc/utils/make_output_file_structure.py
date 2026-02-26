import os
from datetime import datetime
from pathlib import Path
import yaml
import sys


def make_output_file_structure(loop_framework, backend_type, base_path, base_corgiloop_path, final_filename):

    if backend_type=='cgi-howfsc':
        optical_model_type = 'compact_model'
    elif backend_type=='corgihowfsc':
        optical_model_type = 'corgisim_model'
    else: raise NotImplementedError

    base_output_path = os.path.join(base_path, base_corgiloop_path, f'{loop_framework}_gitl')
    os.makedirs(base_output_path, exist_ok=True)

    current_datetime = datetime.now()
    output_folder_name = f'{current_datetime.strftime('%Y-%m-%d_%H%M%S')}_{optical_model_type}'

    fileout_path = os.path.join(base_output_path, output_folder_name, final_filename)
    return fileout_path


def save_run_config(args, fileout):
    """
    Save argparse Namespace (or dict) to a YAML file
    next to the provided output file.

    Args:
        args: argparse.Namespace or dict
        fileout: path to your main output file

    Returns:
        config_path (Path)
    """
    # convert args → dict safely
    cfg = vars(args).copy() if not isinstance(args, dict) else args.copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ensure Path object
    fileout = Path(fileout)

    # build config path (same name, different extension)
    config_path = fileout.parent / "config.yml"
    # add metadata
    cfg["_meta"] = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "fileout": str(fileout),
    }

    # save yaml
    with config_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return config_path


def update_yml(path, updates: dict):
    path = Path(path)
    existing = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    def merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)
            else:
                d[k] = v
        return d

    merged = merge(existing, updates)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
