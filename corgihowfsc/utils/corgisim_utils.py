from functools import lru_cache

import numpy as np
import cgisim

# Mapping configuration:
# cgisim subband labels (per-coronagraph valid 'bandpass' args for CorgiOptics). The wavelength range for each label is
# loaded at runtime from cgisim's own bandpass data (see _get_corgisim_subbands) so the ranges can never drift from what
# cgisim/CorgiOptics actually validates against. Only the label list itself lives here.
CORGISIM_SUBBAND_LABELS = (
    '1a', '1b', '1c',
    '2a', '2b', '2c',
    '3a', '3b', '3c', '3d', '3e', '3g',
    '4a', '4b', '4c',
)

@lru_cache(maxsize=1)
def _get_corgisim_subbands():
    """
    Return {label: (minlam_um, maxlam_um)} for each cgisim subband label.

    Ranges are read from cgisim's own bandpass data via ``cgisim.cgisim_read_bandpass`` (the same data CorgiOptics
    validates against), so they cannot deviate. The result is cached after the first call.

    Returns
    -------
    dict
        Mapping of subband label (e.g. '2a', '3b') to a ``(minlam_um,
        maxlam_um)`` tuple of the band edges in microns.
    """
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'
    subbands = {}
    for label in CORGISIM_SUBBAND_LABELS:
        bp = cgisim.cgisim_read_bandpass(label, info_dir)
        subbands[label] = (bp['minlam_um'], bp['maxlam_um'])
    return subbands

_MANAGER_KEYS = frozenset({
    'bandpass',
    'is_noise_free',
    'Vmag',
    'sptype',
    'ref_flag',
})

CGI_TO_CORGI_MAPPING = {
    'narrowfov': 'hlc', # removed? 
    'nfov_flat': 'hlc', # removed?
    'nfov_dm': 'hlc', # removed? 
    'nfov_band1': 'hlc',
    'spec_band2': 'spc-spec_band{bandpass}', 
    'spec_band3': 'spc-spec_band{bandpass}',
    'wfov_band4': 'spc-wide_band{bandpass}', 
    'specrot_band2': 'spc-spec_band{bandpass}_rotated', 
    'specrot_band3': 'spc-spec_band{bandpass}_rotated', 
    'wfov_band1': 'spc-wide_band{bandpass}',
}

EXPECTED_BANDPASS = {
    'nfov_band1': '1',
    'spec_band2': '2',
    'spec_band3': '3',
    'wfov_band1': '1',
    'wfov_band4': '4',
    'specrot_band2': '2',
    'specrot_band3': '3',
}

SUPPORTED_CGI_MODES = list(CGI_TO_CORGI_MAPPING.keys())
SUPPORTED_CORGI_MODES = list(set(CGI_TO_CORGI_MAPPING.values()))

def _extract_host_properties_from_hconf(hconf):
    """Extract host star properties from hconf object"""
    try:
        star_config = hconf.get('star', {}) if isinstance(hconf, dict) else getattr(hconf, 'star', {})
        
        # Extract stellar properties, preferring target values if available
        Vmag = star_config.get('stellar_vmag')

        sptype = star_config.get('stellar_type')

        return {
            'Vmag': Vmag,
            'spectral_type': sptype,
            'magtype': 'vegamag',  # standard default
            'ref_flag': False  # standard default
        }
    except (AttributeError, KeyError) as e:
        raise ValueError(f"hconf missing required star configuration: {e}")

def map_cgi_to_corgisim_mode(cgi_mode, bandpass): 
    if cgi_mode not in CGI_TO_CORGI_MAPPING:
        raise ValueError(f"Unsupported CGI mode: {cgi_mode}.")

    expected = EXPECTED_BANDPASS.get(cgi_mode)
    if expected is not None and bandpass != expected:
        raise ValueError(
            f"{cgi_mode} requires bandpass {expected}, "
            f"but the cfg wavelength resolved to bandpass {bandpass}"
        )
    return CGI_TO_CORGI_MAPPING[cgi_mode].format(bandpass=bandpass)

def map_wavelength_to_corgisim_subband(wavelength_m):
    """
    Map a wavelength to the cgisim subband label whose range contains it.

    Finds the subband (e.g. '2a', '3b') that actually spans the wavelength. The
    returned label is a valid ``bandpass`` argument for ``CorgiOptics``, and its
    first character is the band number.

    Args:
        wavelength_m: Wavelength in meters.

    Returns:
        Subband label string (e.g. '2a', '3b').

    Raises:
        ValueError: If the wavelength falls in no defined cgisim subband.
    """
    wl_um = wavelength_m * 1e6
    subbands = _get_corgisim_subbands()

    hits = [(label, lo, hi) for label, (lo, hi) in subbands.items()
            if lo <= wl_um <= hi]
    if not hits:
        available = {label: rng for label, rng in subbands.items()}
        raise ValueError(
            f"Wavelength {wl_um*1e3:.1f} nm falls in no CorgiSim subband. "
            f"Available subbands (um): {available}"
        )

    # Subband ranges can overlap slightly at their edges (e.g. 3a ends at
    # 0.693 um, 3b starts at 0.692 um). Break ties by choosing the subband
    # whose center is nearest the wavelength.
    label, _, _ = min(hits, key=lambda h: abs(wl_um - 0.5 * (h[1] + h[2])))
    return label

def calculate_mas_per_lamD(wavelength_m):
    """
    Calculate milliarcseconds per λ/D.
    
    Args:
        wavelength_m: Wavelength in meters
        telescope_diameter_m: Telescope diameter in meters

    """
    D = 2.363114 # Telescope diameter in meters
    theta_rad = wavelength_m / D  # radians
    theta_mas = theta_rad * (180/np.pi) * 3600 * 1000  # convert to mas
    
    return theta_mas