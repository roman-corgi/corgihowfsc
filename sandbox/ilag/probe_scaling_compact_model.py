# Imports and Path Configuration
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Compact model
import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from howfsc.model.mode import CoronagraphMode

howfsc_root = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
model_dir = os.path.join(howfsc_root, 'model')
probes_dir = os.path.join(model_dir, "probes")

# Flat DM for the reference e_field
dm1_flat_path = os.path.join(model_dir, 'dm1', 'dm_allzeros.fits')
dm2_flat_path = os.path.join(model_dir, 'dm2', 'dm_allzeros.fits')

# Arguments
args = get_args(
    niter=1,
    mode='nfov_band1',  # Adjust if working in another band
    precomp='precomp_jacs_always',
    jacpath=os.path.join(os.path.dirname(howfsc_root), 'temp'), # Temp path
    fileout=os.path.join(os.getcwd(), 'output_debug')
)
_, cfgfile, _, _, _, _, _, _ = load_files(args, howfsc_root)
cfg = CoronagraphMode(cfgfile)

# load DM
dm1_flat = fits.getdata(dm1_flat_path)
dm2_flat = fits.getdata(dm2_flat_path)

# Load probes (here the single actuator probe for e.g.)
probe_filename_0 = 'nfov_dm_dmrel_4_1.0e-05_gaussian0.fits'
probe_filename_1 = 'nfov_dm_dmrel_4_1.0e-05_gaussian1.fits'
probe_filename_2 = 'nfov_dm_dmrel_4_1.0e-05_gaussian2.fits'

probe_path_0 = os.path.join(probes_dir, probe_filename_0)
probe_path_1 = os.path.join(probes_dir, probe_filename_1)
probe_path_2 = os.path.join(probes_dir, probe_filename_2)

amp = 0.5
ni_target = 5e-7
wvln_index = 0

# Load probe data once
probe_data_0 = fits.getdata(probe_path_0)
probe_data_1 = fits.getdata(probe_path_1)
probe_data_2 = fits.getdata(probe_path_2)

sl = cfg.sl_list[wvln_index]

# Compute reference fields once
dmlist_ref = [dm1_flat, dm2_flat] # Flat DMs from Cell 2
edm0_ref = sl.eprop(dmlist_ref)
E_lyot_ref = sl.proptolyot(edm0_ref)
efield_ref = sl.proptodh(E_lyot_ref)

# Extract DH mask
out_mask_idx = np.where(np.abs(efield_ref)**2 < 1e-15)
mask = np.ones_like(efield_ref).astype('bool')
mask[out_mask_idx] = False

# Plot DH mask
plt.figure()
plt.imshow(mask, cmap='Greys_r')
plt.title('Extracted DH mask')
plt.colorbar()
plt.show()

# ============================================================
# ITERATIVE LOOP TO ADJUST amp UNTIL cp0 ≈ ni_target
# ============================================================
tolerance = 1e-6
max_iterations = 100
iteration = 0

while iteration < max_iterations:
    # ---- Apply amp to probes ----
    probe_cmd_0 = probe_data_0 * amp
    probe_cmd_1 = probe_data_1 * amp
    probe_cmd_2 = probe_data_2 * amp

    # Apply probes to DM1 (not DM2 /!\)
    dm1_with_probe0 = dm1_flat + probe_cmd_0
    dmlist_probe0 = [dm1_with_probe0, dm2_flat]

    dm1_with_probe1 = dm1_flat + probe_cmd_1
    dmlist_probe1 = [dm1_with_probe1, dm2_flat]

    dm1_with_probe2 = dm1_flat + probe_cmd_2
    dmlist_probe2 = [dm1_with_probe2, dm2_flat]

    # ---- Propagation ----
    edm0_probe0 = sl.eprop(dmlist_probe0)
    E_lyot_total0 = sl.proptolyot(edm0_probe0)

    edm0_probe1 = sl.eprop(dmlist_probe1)
    E_lyot_total1 = sl.proptolyot(edm0_probe1)

    edm0_probe2 = sl.eprop(dmlist_probe2)
    E_lyot_total2 = sl.proptolyot(edm0_probe2)

    # ---- DH E-field ----
    efield_probed0 = sl.proptodh(E_lyot_total0)
    efield_probed1 = sl.proptodh(E_lyot_total1)
    efield_probed2 = sl.proptodh(E_lyot_total2)

    efield_deltaprobe0 = efield_probed0 - efield_ref
    efield_deltaprobe1 = efield_probed1 - efield_ref
    efield_deltaprobe2 = efield_probed2 - efield_ref

    # ---- Delta probe intensities ----
    cp0 = np.nanmean(np.abs(efield_deltaprobe0[mask])**2)
    cp1 = np.nanmean(np.abs(efield_deltaprobe1[mask])**2)
    cp2 = np.nanmean(np.abs(efield_deltaprobe2[mask])**2)

    # Print intermediate cp0 to stdout
    print(cp0)

    # ---- Check convergence ----
    relative_error = abs(cp0 - ni_target) / ni_target
    if relative_error < tolerance:
        break

    # ---- Adjust amp proportionally (cp0 ∝ amp²) ----
    amp = amp * np.sqrt(ni_target / cp0)

    iteration += 1

# ============================================================
# PRINT FINAL RESULTS PROMINENTLY
# ============================================================
print("\n" + "="*60)
print(f"FINAL RESULTS in wvln band index {wvln_index} :")
print(f"cp0: {cp0}")
print(f"cp1: {cp1}")
print(f"cp2: {cp2}")
print(f"ni_target: {ni_target}")
print(f"amp: {amp}")
print("="*60 + "\n")

# ============================================================
# PLOTTING
# ============================================================

# Delta_E to isolate probe impact in Lyot plane
Delta_E0 = E_lyot_total0 - E_lyot_ref
Delta_E1 = E_lyot_total1 - E_lyot_ref
Delta_E2 = E_lyot_total2 - E_lyot_ref

rows = 2
cols = 3
i = 1

vmin_ls = 1e-20
vmax_ls = 3e-6

vmin_probe = ni_target / 10
vmax_probe = ni_target * 1.5

bbmin = 43
bbmax = 110

plt.figure(figsize=(18, 8))

plt.subplot(rows, cols, i)
plt.imshow(np.abs(Delta_E0)**2, cmap='inferno', norm=LogNorm(vmin=vmin_ls, vmax=vmax_ls))
i = i + 1

plt.subplot(rows, cols, i)
plt.imshow(np.abs(Delta_E1)**2, cmap='inferno', norm=LogNorm(vmin=vmin_ls, vmax=vmax_ls))
i = i + 1

plt.subplot(rows, cols, i)
plt.imshow(np.abs(Delta_E2)**2, cmap='inferno', norm=LogNorm(vmin=vmin_ls, vmax=vmax_ls))
i = i + 1

plt.subplot(rows, cols, i)
plt.imshow(np.abs(efield_deltaprobe0)[bbmin:bbmax, bbmin:bbmax]**2, cmap='inferno', norm=LogNorm(vmin=vmin_probe, vmax=vmax_probe))
plt.title(f'DH contrast: {cp0}')
plt.colorbar()
i = i + 1

plt.subplot(rows, cols, i)
plt.imshow(np.abs(efield_deltaprobe1)[bbmin:bbmax, bbmin:bbmax]**2, cmap='inferno', norm=LogNorm(vmin=vmin_probe, vmax=vmax_probe))
plt.title(f'DH contrast: {cp1}')
plt.colorbar()
i = i + 1

plt.subplot(rows, cols, i)
plt.imshow(np.abs(efield_deltaprobe2)[bbmin:bbmax, bbmin:bbmax]**2, cmap='inferno', norm=LogNorm(vmin=vmin_probe, vmax=vmax_probe))
plt.title(f'DH contrast: {cp2}')
plt.colorbar()

plt.show()