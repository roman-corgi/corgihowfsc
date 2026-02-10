# Hardware Configuration File Documentation

## Overview 

Hardware Configuration (`hconf`) files defines parameters used to adjust hardware settings during HOWFSC operations. These YAML-foratted files specify settings for the hardware, including the probe height paramters. 

## Parameter Structure 

### overhead
**Purpose**: setting overhead times for onboard activities
**Unit**: second 

**Example entry**:
```yaml
overhead:
  overdm: 5.0 # overhead with every DM move (only)
  overfilt: 60.0 # overhead with every CFAM move (only)
  overboth: 2.0 # overhead with each DM/CFAM combo (e.g. camera settings)
  overfixed: 5.0 # fixed overhead per iteration for one-offs (e.g. set DM2)
  overframe: 2.0 # overhead per frame for readout
```

### star

**Purpose**: Set the stellar properties for HOWFSC target (used to calculate the exposure time)

**Example entry**:
```yaml
excam:
  cleanrow: 1024
  cleancol: 1024
  scale_method: percentile # must be either 'mean' or 'percentile'
  scale_percentile: 70 # ignored if method is 'mean'
  scale_bright_method: percentile # must be either 'mean' or 'percentile'
  scale_bright_percentile: 99 # ignored if method is 'mean'
```

### excam 
**Purpose**: setting up the EXCAM detector parameter and camera setting constraints

**Example entry**:
```yaml
excam:
  cleanrow: 1024
  cleancol: 1024
  scale_method: percentile # must be either 'mean' or 'percentile'
  scale_percentile: 70 # ignored if method is 'mean'
  scale_bright_method: percentile # must be either 'mean' or 'percentile'
  scale_bright_percentile: 99 # ignored if method is 'mean'
```

### hardware 

**Purpose**: Set up observations

**Example entry**:
```yaml
# Hardware configuration for observation
hardware:
  sequence_list: [CGI_SEQ_NFOV_UNOCC_ASTROM_PHOTOM_PS_1A, CGI_SEQ_NFOV_UNOCC_ASTROM_PHOTOM_PS_1B, CGI_SEQ_NFOV_UNOCC_ASTROM_PHOTOM_PS_1C]
  sequence_observation: CGI_SEQ_NFOV_UNOCC_ASTROM_PHOTOM_PS_1
  pointer: pointer_howfsc.yaml
```

### howfsc

**Purpose**: set up the condition for the WFS&C solver, for instance, clipping bad values and setting up the threshold of the solver. 


**Example entry**:
```yaml
howfsc:
  method: cholesky # tool for least-squares solver
  min_good_probes: 3 # num good probe intensity estimates required per pix
  eestclip: 0.1 # if iinc < -icoh*eestclip, e-field marked as bad
  eestcondlim: 0.4 # if lstsq solve has cond number below this, e-field bad
```

### probe

**Type**: a list of probe height values for a relative DM setting with scale 1

**Purpose**: ?? TBC

**Example entry**:

```yaml
probe:
  dmrel_ph_list: [1.0e-5, 1.0e-5, 1.0e-5]
```