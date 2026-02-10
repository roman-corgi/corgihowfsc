# Control Strategy File Documentation

## Overview

Control strategy files define the parameters used to adjust wavefront control performance 
during HOWFSC (High-Order Wavefront Sensing and Control) operations. These YAML-formatted 
files specify iteration-dependent settings that control the Electric Field Conjugation (EFC) 
algorithm behavior.

## File Format

Control strategy files use YAML format with a specific structure for parameter definitions. 
Each parameter can be defined for different iteration ranges and contrast regimes.

### Parameter Structure

Each parameter consists of one or more entries with the following fields:

- **first**: First iteration number where this setting applies (1-indexed)
- **last**: Last iteration number where this setting applies (use `None` for infinity, use last==first if only applicable for a single iteration)
- **low**: Lower bound (closest to zero) of mean total contrast where this setting applies (0 for all)
- **high**: Upper bound (furthest from zero) of mean total contrast where this setting applies (use `None` for infinity)
- **value**: The parameter value to use in this range where low<=contrast<high and first<=iteration<last

Multiple entries allow parameters to change based on iteration number and achieved contrast level.

---

## Parameter Definitions

### regularization

**Type**: log10 of the value relative to square of the largest singular value of the
weighted Jacobian. 

**Purpose**: Used to vary the allowable strength of the EFC correction. Values closer to -infinity allow for larger DM stroke. -5 is considered agressive and -2 is mild.

**Units**: n/a

**Typical Range**: -2 -- -5

**Details**:

**Example Entry**:
```yaml
regularization:
  - first: 1          # Iterations 1-4
    last: 4
    low: 0           # All contrast levels better than 1e-5
    high: 1.0e-5
    value: -2.0      # Mild stroke permitted
  - first: 5          # Iteration 5 only
    last: 5
    low: 0   
    high: 1.0e-5     # All contrast levels better than 1e-5
    value: -5.0      # Large stoke allowed for one iteration
```

**Notes**:

---

### pixelweights

**Type**: Relative filepath string

**Purpose**: 

**File Format**: FITS file containing N 2D weighting matrices, the first in its primary HDU and the remaining N-1 2D 
weighting sequentially in image HDUs. There should be one for each wavelength in an optical model, running from shortest 
to longest wavelength in order. 

**Details**: Unweighted pixels should have weight = 1; weighting a pixel by Y implies that control 
will value the reduction in intensity at that pixel by Y^2.

**Example Entry**:
```yaml
pixelweights:
  - first: 1
    last: None       # Applied to all iterations
    low: 0
    high: None      # Applied for all contrasts
    value: '../../every_mask_config/pixelweights_ones_nlam3_nrow153.fits'
```

**File Requirements**:
- Shape: [nlam, nrow, ncol]
- Valid values: 1.0


**Notes**:

[//]: # (- [FILL IN: When to use uniform weights vs. custom weights])

[//]: # (- [FILL IN: How to generate custom pixel weight files])

---

### dmmultgain

**Type**: Float

**Purpose**: Adjusts the output of EFC using the multiplicative gain provided in the value

**Units**: n/a

**Typical Range**: 0.5-2

**Details**: A dmmultgain = 1 does not change the relative DM setting
produced by wavefront control. A dmmultgain of 0.5 reduces the
applied DM change by half.

**Example Entry**:
```yaml
dmmultgain:
  - first: 1
    last: None
    low: 0
    high: None
    value: 1         # Use EFC output as-is for all scenarios
```

**Notes**:

---

### unprobedsnr

**Type**: Float

**Purpose**: Signal-to-noise ratio target for unprobed measurements

**Units**: SNR

**Typical Range**: ~5, must be >0

**Details**: 

**Example Entry**:
```yaml
unprobedsnr:
  - first: 1
    last: None
    low: 0
    high: None
    value: 5         # 
```

**Notes**:
- Needs to be high enough to have a good estimate of the contrast since the contrast is used to choose exposure times etc

---

### probedsnr

**Type**: Float

**Purpose**: Signal-to-noise ratio target for probed measurements

**Units**: SNR

**Typical Range**: ~7, must be >0

**Details**: 

[//]: # ([FILL IN: Explain probed SNR in the context of pair-wise probing,)
[//]: # (why it differs from unprobed SNR, and how it affects field estimation accuracy])

**Example Entry**:
```yaml
probedsnr:
  - first: 1
    last: None
    low: 0
    high: None
    value: 7         # 
```

**Notes**:


---

### probeheight

**Type**: Float

**Purpose**: Amplitude of DM probes for pair-wise probing

**Units**: mean probe amplitude

**Typical Range**: 1e-5--1e-7, must be >0

**Details**: 

**Example Entry**:
```yaml
probeheight:
  - first: 1
    last: None
    low: 0           # High contrast regime
    high: 1.0e-7
    value: 1.0e-7    # 
  - first: 1
    last: None
    low: 1.0e-7      # Medium contrast
    high: 1.0e-6
    value: 1.0e-6    # 
  - first: 1
    last: None
    low: 1.0e-6      # Low contrast regime
    high: None
    value: 1.0e-5    # 
```

**Notes**:
- Non-linearity concerns at deep contrast
- Probe height relates to achievable contrast floor
- Relationship to SNR requirements

---

### fixedbp

**Type**: File path (single value, not range-based)

**Purpose**: Specifies fixed bad pixels to exclude

**File Format**: FITS file containing binary mask

**Details**: Bad pixels are identified as 1.

**Example Entry**:
```yaml
fixedbp: '../../every_mask_config/fixedbp_zeros.fits'
```

**File Requirements**:
- Shape: [nrow, ncol]
- Valid values: 0 = good, 1 = bad

**Notes**:

---

## Usage Guidelines

[//]: # (### Creating a New Control Strategy)

[//]: # ()
[//]: # ([FILL IN: Step-by-step process for creating a new control strategy file])

[//]: # ()
[//]: # (1. [FILL IN])

[//]: # (2. [FILL IN])

[//]: # (3. [FILL IN])

[//]: # ()
[//]: # (### Modifying Existing Strategies)

[//]: # ()
[//]: # ([FILL IN: Best practices for adjusting parameters])

[//]: # ()
[//]: # (- [FILL IN: Which parameters to adjust first])

[//]: # (- [FILL IN: Testing procedures])

[//]: # (- [FILL IN: Common modifications for different scenarios])

[//]: # ()
[//]: # (### Iteration and Contrast Logic)

[//]: # ()
[//]: # (The control strategy parameters are selected based on:)

[//]: # (```python)

[//]: # (# Pseudocode for parameter selection)

[//]: # (for each_iteration:)

[//]: # (    current_contrast = measure_current_contrast&#40;&#41;)

[//]: # ()
[//]: # (    for each_parameter:)

[//]: # (        # Find matching entry)

[//]: # (        matching_entry = find_where&#40;)

[//]: # (            iteration_number >= first AND)

[//]: # (            iteration_number <= last AND)

[//]: # (            current_contrast >= low AND)

[//]: # (            current_contrast < high)

[//]: # (        &#41;)

[//]: # (        )
[//]: # (        use_parameter_value = matching_entry.value)

[//]: # (```)

[//]: # ()
[//]: # ([FILL IN: Explain what happens if multiple entries match, or if no entries match])

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Common Control Strategy Configurations)

[//]: # ()
[//]: # (### Conservative Strategy &#40;High Stability&#41;)

[//]: # ()
[//]: # ([FILL IN: Parameter recommendations for stable but slower convergence])

[//]: # ()
[//]: # (### Aggressive Strategy &#40;Fast Convergence&#41;)

[//]: # ()
[//]: # ([FILL IN: Parameter recommendations for faster convergence with potential stability trade-offs])

[//]: # ()
[//]: # (### Deep Contrast Strategy)

[//]: # ()
[//]: # ([FILL IN: Parameter recommendations for achieving deepest possible contrast])

[//]: # ()
[//]: # (---)

## File Path Conventions

Relative paths in control strategy files are resolved relative to the location of the cstrat file.

[//]: # (The example paths shown &#40;`../../every_mask_config/`&#41; suggest [FILL IN: explain directory structure].)


---

## References
- Cady et al. (2025) - CGI HOWFSC architecture and performance

---

## Change History
- 01/21/2026 - Initial documentation created
