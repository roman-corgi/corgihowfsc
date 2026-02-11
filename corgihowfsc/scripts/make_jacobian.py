import os
from pathlib import Path
from datetime import datetime
from howfsc.scripts.jactest_mp import calculate_jacobian_multiprocessed

base_path = Path.home()  # this is the proposed default but can be changed
base_corgiloop_path = 'corgiloop_data'
jacobian_path = os.path.join(base_path, base_corgiloop_path, 'jacobians')
os.makedirs(jacobian_path, exist_ok=True)

current_datetime = datetime.now()
jacobian_file_path = f"{jacobian_path}/jacobian_{current_datetime.strftime('%Y-%m-%d_%H%M%S')}.fits"

calculate_jacobian_multiprocessed(output=jacobian_file_path, proc=0)
