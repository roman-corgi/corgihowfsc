import os
from datetime import datetime


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
