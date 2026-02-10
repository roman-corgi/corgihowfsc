import os
from pathlib import Path
from datetime import datetime
from howfsc.scripts.jactest_mp import calculate_jacobian_multiprocessed

home_directory = Path.home()
jacobian_path = os.path.join(home_directory,'cpp_data','jacobians')
os.makedirs(jacobian_path, exist_ok=True)

current_datetime = datetime.now()
jacobian_file_path = f"{jacobian_path}/jacobian_{current_datetime.strftime('%Y-%m-%d_%H%M%S')}.fits"

calculate_jacobian_multiprocessed(output=jacobian_file_path, proc=0)
