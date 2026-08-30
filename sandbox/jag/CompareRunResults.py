import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from astropy.io import fits
from corgisim import scene, instrument

plt.rcParams.update({'font.size': 14})

import roman_preflight_proper

roman_preflight_proper.copy_here()

comparison = 'Individual'   
                            # Options:  'LSandFSAM' to look at the tiny differences the signs cause
                            #           'Individual' to look at each of the effects on its own
                            #           'LeastSignificantErrors' to look at the effects that have the least impact
                            #           'MostSignificantErrors' to look at the errors that have the most impact
                            #           'Star_on_FSM_options' to compare options for the star on FPM alignment (~2.7 mas) (doesn't really make any difference)

if comparison == 'LSandFSAM':
    folder_names = {
        'LS++,FS++' : 's26_AlltoDate_LS++_FS++/2026-06-18_183437_corgisim_model',
        'LS++,FS+-' : 's26.1_AlltoDate_LS++_FS+-/2026-06-18_192216_corgisim_model',
        'LS++,FS-+' : 's26.2_AlltoDate_LS++_FS-+/2026-06-18_201228_corgisim_model',
        'LS++,FS--' : 's26.3_AlltoDate_LS++_FS--/2026-06-18_205952_corgisim_model',
        'LS+-,FS++' : 's26.4_AlltoDate_LS+-_FS++/2026-06-19_085727_corgisim_model',
        'LS+-,FS+-' : 's26.5_AlltoDate_LS+-_FS+-/2026-06-19_095555_corgisim_model',
        'LS+-,FS-+' : 's26.6_AlltoDate_LS+-_FS-+/2026-06-19_104955_corgisim_model',
        'LS+-,FS--' : 's26.7_AlltoDate_LS+-_FS--/2026-06-19_115117_corgisim_model'
        }
elif comparison == 'Individual':
    folder_names = {
        #'All errors combined': 's26_AlltoDate_LS++_FS++/2026-06-18_183437_corgisim_model',
        'All errors combined': 's29_AlltoDate/2026-06-29_101445_corgisim_model',
        'PMN Creep': 's2_Creep_without_beta_bumping/2026-05-22_085333_corgisim_model',
        'DM Hysteresis' : 's3_Hysteresis_without_beta_bumping/2026-05-22_094445_corgisim_model',
        '30 nm Focus on primary': 's6_30nmFocus_only/2026-06-08_134629_corgisim_model',
        '2.5 nm Z6 on DM1': 's8_Z6onDM1/2026-06-16_131553_corgisim_model',
        '2.5 nm Z6 on DM2': 's9_Z6onDM2/2026-06-16_140239_corgisim_model',
        'LS Misalignment': 's13_LSmisalignment_++/2026-06-18_131144_corgisim_model',
        'FSAM Misalignment': 's22_FieldStopMisalignment_++/2026-06-17_204726_corgisim_model',
        '~2.7 mas Source shift': 's27.2_SourceShift_+x_+y/2026-06-25_142718_corgisim_model',
        'No errors' : 's0_perfect/2026-06-16_121105_corgisim_model'
        }
elif comparison == 'LeastSignificantErrors':
    folder_names = {
        'DM Hysteresis' : 's3_Hysteresis_without_beta_bumping/2026-05-22_094445_corgisim_model',
        '30 nm Focus on primary': 's6_30nmFocus_only/2026-06-08_134629_corgisim_model',
        'LS Misalignment': 's13_LSmisalignment_++/2026-06-18_131144_corgisim_model',
        'FSAM Misalignment': 's22_FieldStopMisalignment_++/2026-06-17_204726_corgisim_model',
        'Perfect case' : 's0_perfect/2026-06-16_121105_corgisim_model'
        }
elif comparison == 'MostSignificantErrors':
    folder_names = {
        'PMN Creep': 's2_Creep_without_beta_bumping/2026-05-22_085333_corgisim_model',
        '2.5 nm Z6 on DM1': 's8_Z6onDM1/2026-06-16_131553_corgisim_model',
        '2.5 nm Z6 on DM2': 's9_Z6onDM2/2026-06-16_140239_corgisim_model',
        '2.5 nm Z6 on DM1 and DM2': 's10_Z6onDM1and2_only/2026-06-16_145736_corgisim_model',
        'PMN Creen and 2.5 nm Z6 on DM1 and DM2':'s11_Z6onDM1and2_Creep/2026-06-16_155213_corgisim_model',
        'All errors considered to date': 's26_AlltoDate_LS++_FS++/2026-06-18_183437_corgisim_model'
        }
elif comparison == 'Star_on_FSM_options':
    folder_names = {
        'source shift x':'s27.0_SourceShiftx/2026-06-25_105057_corgisim_model',
        'source shift y':'s27.1_SourceShifty/2026-06-25_123421_corgisim_model',
        'source shift +x +y':'s27.2_SourceShift_+x_+y/2026-06-25_142718_corgisim_model',
        'source shift +x -y':'s27.3_SourceShift_+x_-y/2026-06-25_152014_corgisim_model',
        'FPM shift x':'s28.0_FPMshiftx/2026-06-26_140254_corgisim_model',
        'FPM shift y':'s28.1_FPMshifty/2026-06-26_145856_corgisim_model',
        'FPM shift +x +y':'s28.2_FPMshift_+x_+y/2026-06-26_155046_corgisim_model',
        'FPM shift +x -y':'s28.3_FPMshift_+x_-y/2026-06-26_163932_corgisim_model'}


home_path = os.path.join('/Users/jessicag','corgiloop_data/corgi-howfsc_gitl')

base_paths = {}
for key in folder_names.keys():
    base_paths[key] = os.path.join(home_path, folder_names[key])


cmap = plt.get_cmap("tab20b")
linestyles = ["-", "--", "-.", ":"]

fig, ax = plt.subplots(figsize=(9, 5))

for i, (label, base_path) in enumerate(base_paths.items()):
    base = os.path.normpath(base_path)
    csv_path = os.path.join(base, "measured_contrast.csv")

    if not os.path.isfile(csv_path):
        print(f"WARNING: not found – {csv_path}")
        continue

    contrast = pd.read_csv(csv_path, skiprows=1, header=None).squeeze()
    iters = np.arange(len(contrast))+1

    ax.plot(iters, contrast, marker="o", markersize=3, linewidth=1.5,
            color=cmap(i / max(len(base_paths) - 1, 1)),
            linestyle=linestyles[i % len(linestyles)],
            label=label)

ax.set_xlabel("Iteration")
ax.set_ylabel("Measured Contrast")
ax.set_title("Measured Contrast vs. Iteration (HLC B1)\ncorgihowfsc full model PWP estimator")
ax.set_yscale("log")
ax.set_xlim([0,11])
ax.grid(True, which="both", alpha=0.3)
# ax.legend(fontsize=14, loc="upper right")
ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
plt.tight_layout()
plt.show()

