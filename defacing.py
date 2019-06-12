import os
import subprocess
from joblib import Parallel, delayed

all_files = os.listdir('IXI-T1')

def deface_nii(all_files, i):
    each_file = all_files[i]
    each_file_path = os.path.join('IXI-T1',each_file)
    subprocess.call(['pydeface',each_file_path])

Parallel(n_jobs=-1)(delayed(deface_nii)(all_files, i) for i in range(len(all_files)))