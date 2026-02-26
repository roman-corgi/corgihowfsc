import numpy as np

# Mapping configuration - easy to update for new modes
# NOTE - Add new mappings here as support is added

CGI_TO_CORGI_MAPPING = {
    'narrowfov': 'hlc',
    'nfov_flat': 'hlc', 
    'nfov_dm': 'hlc',
    'nfov_band1': 'hlc',
    'spec_band2': 'spc-spec_band2', 
    'spec_band3': 'spc-spec_band3',
    'wfov_band4': 'spc-wide',
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

# Helper function to map wavelength to corgisim bandpass
def map_wavelength_to_corgisim_bandpass(wavelength_m, tolerance=3e-9):
    """
    Map wavelength to CorgiSim bandpass label.
    
    Args:
        wavelength_m: Wavelength in meters
        tolerance: Matching tolerance in meters (default ±3nm)
        
    Returns:
        CorgiSim bandpass label ('1', '2', '3', or '4')
    """
    corgisim_wavelengths = {
        '1': 575e-9, # 1B
        '2': 638e-9, # 2B
        '3': 727e-9, # 3C
        '4': 825e-9} # 4B
    
    for bandpass, wl in corgisim_wavelengths.items():
        if abs(wavelength_m - wl) <= tolerance:
            return bandpass
    
    available_nm = [wl * 1e9 for wl in corgisim_wavelengths.values()]
    raise ValueError(f"Wavelength {wavelength_m*1e9:.1f} nm does not match any "
                    f"CorgiSim options {available_nm} nm within ±{tolerance*1e9:.0f} nm")

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