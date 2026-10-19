"""
Wrapper script to run GITL DM seed generation for NFOV Band 1 both sides configuration.
This script simply calls the main run_corgisim_nulling_gitl.py with the appropriate parameter file.
"""
import os
import sys

# Add the scripts directory to the path so we can import run_corgisim_nulling_gitl
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
sys.path.insert(0, scripts_dir)

from run_corgisim_nulling_gitl import main


if __name__ == '__main__':
    
    # Set the default parameter file for NFOV Band 1 both sides configuration
    param_file_name = os.path.abspath(os.path.join('..', 'model', 'nfov_band1', 'nfov_band1_both_sides', 'params_tvac_plus_50_compact.yaml'))
    main(param_file_name=param_file_name, fullpath=True)
