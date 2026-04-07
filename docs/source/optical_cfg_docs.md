# Optical Model Configuration File Documentation

## Overview

Optical model configuration files define the parameters used to initialize the coronagraph optical model (`CoronagraphMode`) for HOWFSC operations. These YAML-formatted files specify the deformable mirror (DM) geometry and voltage settings, initial DM commands, and per-wavelength (subband) optical element configurations including the field stop, focal plane mask, Lyot stop, entrance pupil, and dark hole mask.

All file paths may be absolute or relative. If relative, they are resolved relative to the location of the YAML file itself. When placing a copy of this file outside the repository, all relative paths must be converted to absolute paths.

## Parameter Structure

### dms

**Purpose**: Define the geometry, registration, and voltage constraints for each deformable mirror.

Each DM entry (e.g. `DM1`, `DM2`) contains two subsections: `registration` and `voltages`.

**registration** — optical geometry of the DM:

| Parameter | Description |
|-----------|-------------|
| `dx`, `dy` | DM center offset from nominal, in pixels |
| `inffn` | Path to FITS file containing the actuator influence function |
| `nact` | Number of actuators along one axis (e.g. 48 for a 48x48 DM) |
| `ppact_cx`, `ppact_cy` | Pixels per actuator in x and y |
| `ppact_d` | Pixels per actuator diagonal |
| `thact` | Actuator grid rotation angle, in degrees |
| `flipx` | Whether to flip the DM in x (`true` or `false`) |

**voltages** — DM voltage limits and maps:

| Parameter | Description |
|-----------|-------------|
| `gainfn` | Path to FITS file containing the per-actuator voltage gain map |
| `vmax`, `vmin` | Maximum and minimum allowed voltages, in volts |
| `vneighbor` | Maximum allowed voltage difference between neighboring actuators |
| `vcorner` | Maximum allowed voltage difference between corner-adjacent actuators |
| `vquant` | Voltage quantization step size, in volts |
| `tiefn` | Path to FITS file defining tied (ganged) actuator pairs |
| `flatfn` | Path to FITS file defining the flat (zero) DM command |
| `crosstalkfn` | Path to YAML file defining actuator crosstalk, or `null` if not used |

**z** — axial position of the DM along the optical axis (DM1 = 0.0, DM2 = 1.0 by convention).

**Example entry**:
```yaml
dms:
  DM1:
    pitch: 0.0009906
    registration:
      dx: -0.014224950169164925
      dy: -0.04932028167763836
      inffn: ../../dm1/dm_1236-2_brian_inf_v1.0.fits
      nact: 48
      ppact_cx: 6.436212500311111
      ppact_cy: 6.3393938970505905
      ppact_d: 13
      thact: 90.02875243700015
      flipx: true
    voltages:
      gainfn: ../any/gain_map_dm1.fits
      vmax: 100
      vmin: 0
      vneighbor: 50.0
      vcorner: 75.0
      vquant: 0.001678466796875
      tiefn: ../../dm1/tied_actuator_map.fits
      flatfn: ../../dm1/dm_allzeros.fits
      crosstalkfn: null
    z: 0.0
```

---

### init

**Purpose**: Specify the absolute initial DM voltage command applied at the start of a HOWFSC loop, before any wavefront control updates are applied.

| Parameter | Description |
|-----------|-------------|
| `dminit` | Path to FITS file containing the 48x48 initial DM voltage map, in volts |

**Example entry**:
```yaml
init:
  DM1:
    dminit: ../any/dmabs_init_dm1.fits
  DM2:
    dminit: ../any/dmabs_init_dm2.fits
```

---

### sls

**Purpose**: Define the per-wavelength (subband) optical configuration. Each numbered entry (0, 1, 2, ...) corresponds to one CFAM filter / wavelength channel used in the sensing loop. The number of entries must match the number of wavelengths expected by the control strategy.

| Parameter | Description |
|-----------|-------------|
| `lam` | Central wavelength of this subband, in meters |
| `ft_dir` | Fourier transform direction for propagation; typically `reverse` |

Each subband contains the following optical element subsections:

**sp** — entrance pupil stop:

| Parameter | Description |
|-----------|-------------|
| `afn` | Path to FITS file for the amplitude mask |
| `pfn` | Path to FITS file for the phase mask |
| `pdp` | Pupil diameter in pixels |

**lyot** — Lyot stop:

| Parameter | Description |
|-----------|-------------|
| `afn` | Path to FITS file for the amplitude mask |
| `pfn` | Path to FITS file for the phase mask |
| `pdp` | Pupil diameter in pixels |
| `tip`, `tilt` | Tip and tilt offsets applied at the Lyot plane |

**epup** — entrance pupil phase/amplitude (aberration maps):

| Parameter | Description |
|-----------|-------------|
| `afn` | Path to FITS file for the amplitude map |
| `pfn` | Path to FITS file for the phase map |
| `pdp` | Pupil diameter in pixels |
| `tip`, `tilt` | Tip and tilt offsets |

**fs** — field stop:

| Parameter | Description |
|-----------|-------------|
| `afn` | Path to FITS file for the field stop amplitude mask |
| `pfn` | Path to FITS file for the field stop phase mask |
| `ppl` | Pixels per lambda/D at the field stop plane |

**fpm** — focal plane mask:

| Parameter | Description |
|-----------|-------------|
| `afn` | Path to FITS file for the FPM amplitude mask |
| `pfn` | Path to FITS file for the FPM phase mask |
| `ppl` | Pixels per lambda/D at the focal plane |
| `isopen` | If `true`, the FPM is treated as open (no occulting mask applied) |

**dh** — dark hole mask:

| Parameter | Description |
|-----------|-------------|
| *(value)* | Path to FITS file defining the dark hole pixel mask. Pixels set to 1 are included in wavefront control; pixels set to 0 are excluded. |

**Example entry**:
```yaml
sls:
  0:
    lam: 5.545370626178385e-07
    ft_dir: reverse
    sp:
      afn: ../../every_mask_config/ones_like_pupil.fits
      pdp: 299.1448059082031
      pfn: ../../every_mask_config/zeros_like_pupil.fits
    lyot:
      afn: ../any/lyot_amp.fits
      pdp: 299.1448059082031
      pfn: ../any/subband0/lyot_ph.fits
      tip: 0.0
      tilt: 0.0
    epup:
      afn: ../any/epup_amp.fits
      pdp: 299.1448059082031
      pfn: ../any/subband0/epup_ph.fits
      tip: 0
      tilt: 0
    fs:
      afn: subband0/fs_amp.fits
      pfn: ../../every_mask_config/zeros_like_fs.fits
      ppl: 2.1936521784878877
    fpm:
      afn: ../any/subband0/fpm_amp.fits
      isopen: true
      pfn: ../any/subband0/fpm_ph.fits
      ppl: 11.572947393763585
    dh: subband0/dh_sw_mask.fits
```

---

## Notes on File Paths

All `*fn` and `dh` entries accept either absolute or relative paths. Relative paths are resolved relative to the directory containing the YAML file. When copying this file to a location outside the repository (e.g. to `C:/Users/.../corgiloop_data/alternate_files/`), convert all relative paths to absolute paths to avoid `OSError` at runtime. This affects every `afn`, `pfn`, `gainfn`, `tiefn`, `flatfn`, `crosstalkfn`, `inffn`, `dminit`, and `dh` entry across all DM and subband sections.
