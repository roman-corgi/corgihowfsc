# Probe
## Focal plane wavefront sensing: pairwise probing (PWP)

### Theoretical framework

In the code, the focal plane wavefront sensing uses Pairwise Probing (PWP) to estimate the complex electric field of the coherent starlight speckles. By applying known phase perturbations ( = probes) to the DM, we modulate the coherent stellar flux to extract the speckles electric field.

Let $E_0$ be the unperturbed electric field in the pupil plane, and $C$ the linear coronagraphic propagation operator. Because the host star and the exoplanet are incoherent, the total intensity measured on the detector is the sum of coherent and incoherent intensities:
$$I_0 = |C[E_0]|^2 + I_{incoh}$$

When applying a pair of symmetric probes $\{+, -\}$, we induce a phase shift $\Delta\Psi$ on the DM. Assuming the phase modulation primarily affects the coherent stellar flux (and under the small aberration approximation where the optical path difference is small compared to the wavelength), the measured intensities on the detector are:
$$I_{+} \approx |C[E_0] + i\Delta p|^2 + I_{incoh}$$
$$I_{-} \approx |C[E_0] - i\Delta p|^2 + I_{incoh}$$
where $\Delta p = \frac{C[E_0e^{i\Delta\Psi}]-C[E_0]}{i}=\frac{C[E_0]-C[E_0e^{-i\Delta \Psi}]}{i}$.
In fact, the $\Delta p$ we use is the mean value using both definition above.

Using the theoretical model, we obtain the phase of the probe's electric field by taking the argument of the mean $\Delta_p$.
To obtain the field amplitude, we do not use the model but an empirical measurement: we show that 
$$|\Delta p| \approx \sqrt{\frac{I_{+} + I_{-}}{2} - I_0}$$

To estimate the stellar speckles, we take for the probe number $n$ the difference of the paired images to isolate the interference cross-terms. This fundamental step removes the static unmodulated terms, including the incoherent planetary signal:
$$\delta_n = \frac{I_{+,n} - I_{-,n}}{2} \approx -2\Re(C[E_0])\Im(\Delta p_n) + 2\Im(C[E_0])\Re(\Delta p_n)$$

By applying multiple independent probes (in the code it's 3 pairs $\Delta p_1$, $\Delta p_2$ and $\Delta p_3$ to increase measurement robustness so 6 probes), we can construct an invertible linear system for every pixel:
$$
\begin{bmatrix}
-2\Im(\Delta p_1) & 2\Re(\Delta p_1) \\
-2\Im(\Delta p_2) & 2\Re(\Delta p_2) \\
-2\Im(\Delta p_3) & 2\Re(\Delta p_3)
\end{bmatrix}
\begin{bmatrix}
\Re(C[E_0]) \\
\Im(C[E_0])
\end{bmatrix}
=
\begin{bmatrix}
\delta_1 \\
\delta_2 \\
\delta_3
\end{bmatrix}
$$
## Software implementation 
With corgihowfsc, you can select from several types of probes as soon as you start your simulation. 
### Core sensing interface (`corgihowfsc/sensing/`)

#### `Probes.py`
This file defines the `Probes` class. It establishes the basic methods required for the simulation:
* `get_dm_probes()`: Returns the voltage {$V_{+}, V_{-}$} corresponding to the requested probe {$I_{+}, I_{-}$} 
* `get_probe_ap()`: Computes the analytical complex electric field ($\Delta p$) of the probe in the focal plane. To see the images, go find the result of your simulation to `corgiloop_data/corgi-howfsc_gitl`. For each iterations, you will find `images.fits` which contains, for each subband, one image without a probe and six with a probe.
```
├── config.yml
├── ...
├── iteration_0001
│   ├── dm1_command.fits
│   ├── dm2_command.fits
│   ├── efield_estimations.fits
│   ├── images.fits
│   ├── intensity_coherent.fits
│   ├── intensity_incoherent.fits
│   ├── intensity_total.fits
│   └── perfect_efields.fits
├── iteration_0002
│   └── ...
├── ...
└── iteration_XXXX
    └── ...
```
#### `GettingProbes.py`
This file defines the `ProbesShapes` class. It's a class that inherits from `Probes`. 
It's responsible for reading the pre-computed probe shapes. These are stored as `.fits` files in `corgihowfsc/model/probes"`.

### Generation scripts (`corgihowfsc/scripts/`)

The complex spatial geometries of the probes are synthesized offline before running a loop. These scripts generate the 2D voltage maps that will be loaded by `GettingProbes.py`.

#### `write_sinc_probes.py`
Generates the default "Sinc-Sinc-Sine" probing maps. This is the default shape on board

#### `write_gaussian_probes.py`
An alternative generator that produces Gaussian probes on the DM. They should be the preferred choice of alternative probes for the actual instrument

### Choosing your probe shape for a run (`corgihowfsc/scripts/default_param.yml"`)
Before running a loop, you can choose from 4 different probe shapes `{'default', 'single', 'gaussian', 'unmodulated_sinc'}` by specifying it in the YAML file under the section `sim_settings` using the parameter `probe_shape`:
```yaml
sim_settings:
  loop_framework: "corgi-howfsc" # do not modify 
  precomp: "precomp_jacs_always" # options: 'precomp_jacs_always', 'precomp_all_once', 'load_all' (only if defjacpath is not None)
  output_every_iter: true # whether to save the output frames at every iteration (true) or after all iterations are done (false)
  niter: 3 # number of iterations to run
  mode: "nfov_band1" # options: see models
  dark_hole: "360deg"  # options: see models
  probe_shape: "gaussian"
 ```
Please note: if you want to control the amplitude of your probe, you should be aware that the code automatically applies a multiplier factor `scale_factor_list` depending, in particular, on the current contrast level.
### References
* CADY, Eric, BOWMAN, Nicholas, GREENBAUM, Alexandra Z., et al. High-order wavefront sensing and control for the Roman Coronagraph Instrument (CGI): architecture and measured performance. Journal of Astronomical Telescopes, Instruments, and Systems, 2025, vol. 11, no 2, p. 021408-021408.


* KRIST, John E., STEEVES, John B., DUBE, Brandon D., et al. End-to-end numerical modeling of the Roman Space Telescope coronagraph. Journal of Astronomical Telescopes, Instruments, and Systems, 2023, vol. 9, no 4, p. 045002-045002.