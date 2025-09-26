import numpy as np
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, Moffat2DKernel

def generate_galaxy_component(simulator, filter_band='r'):
    """
    Generates the ideal galaxy component of the image using a Sersic profile.

    Args:
        simulator (LSSTAdvancedSimulator): The main simulator instance.
        filter_band (str): The filter band for the observation.

    Returns:
        ndarray: An image array containing the ideal galaxy.
    """
    # 1. Determine galaxy magnitude (brightness) randomly and convert to amplitude.
    zero_point_mag = simulator.zero_points_30s.get(filter_band, 28.1)
    mag_r = np.random.uniform(18.0, 24.5)
    amplitude = 10**(0.4 * (zero_point_mag - mag_r))

    # 2. Determine galaxy type (elliptical, spiral, irregular) probabilistically.
    galaxy_type_prob = np.random.random()
    if galaxy_type_prob < 0.3:  # Elliptical (30% probability)
        params = {
            'type': 'elliptical', 'n': np.clip(np.random.normal(4.0, 0.5), 0.5, 8.0),
            'r_eff_kpc': np.random.lognormal(np.log(4.0), 0.3),
            'ellip': np.clip(np.random.beta(2, 2) * 0.8, 0.0, 0.9), 'amplitude': amplitude * 1.2
        }
    elif galaxy_type_prob < 0.8:  # Spiral (50% probability)
        params = {
            'type': 'spiral', 'n': np.clip(np.random.normal(1.0, 0.2), 0.5, 8.0),
            'r_eff_kpc': np.random.lognormal(np.log(4.5), 0.4),
            'ellip': np.clip(np.random.beta(1.5, 3) * 0.6, 0.0, 0.9), 'amplitude': amplitude
        }
    else:  # Irregular (20% probability)
        params = {
            'type': 'irregular', 'n': np.random.uniform(0.5, 2.0),
            'r_eff_kpc': np.random.lognormal(np.log(2.5), 0.6),
            'ellip': np.clip(np.random.beta(1, 1) * 0.7, 0.0, 0.9), 'amplitude': amplitude * 0.8
        }

    # 3. Simulate redshift effects to adjust galaxy size.
    redshift_factor = np.random.uniform(0.8, 1.5)
    r_eff_pixels = params['r_eff_kpc'] * 5.0 / redshift_factor

    # 4. Update and clip final parameters to ensure realistic ranges.
    params.update({
        'amplitude': np.clip(params['amplitude'], 100, 5e5),
        'r_eff': np.clip(r_eff_pixels, 1.0, simulator.image_size // 6),
        'theta': np.random.uniform(0, np.pi), # Random orientation
        'x_0': simulator.image_size / 2 + np.random.normal(0, 3.0), # Slightly offset from center
        'y_0': simulator.image_size / 2 + np.random.normal(0, 3.0)
    })

    # Generate the galaxy image using the Sersic2D model.
    allowed_keys = ["amplitude", "r_eff", "n", "x_0", "y_0", "ellip", "theta"]
    sersic_params = {k: v for k, v in params.items() if k in allowed_keys}
    ideal_galaxy = Sersic2D(**sersic_params)(simulator.x, simulator.y)
    
    # Store galaxy type for logging.
    simulator.simulation_metadata['galaxy_type'] = params.get('type', 'unknown')
    
    return ideal_galaxy

def add_stars_component(simulator, image, galactic_latitude=30.0, filter_band='r', add_spikes=True):
    """
    Adds background stars to the image.
    """
    star_image = image.copy()
    
    # Simulate higher star density closer to the galactic plane.
    lat_factor = 1.0 / (np.abs(np.sin(np.deg2rad(galactic_latitude))) + 0.1)
    base_density = 10000
    star_density = base_density * min(lat_factor, 5.0)
    patch_area_deg2 = (simulator.image_size * simulator.pixel_scale / 3600)**2
    expected_stars = star_density * patch_area_deg2
    num_stars = np.random.poisson(expected_stars)
    num_stars = min(num_stars, 200)
    
    # Create PSF kernel for stars.
    seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.15), 0.4, 1.2)
    psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / simulator.pixel_scale / 2.35, alpha=2.8)
    
    for _ in range(num_stars):
        zero_point_mag = 28.1
        mag = np.random.power(2.35) * 18 + 12
        amplitude = 10**(0.4 * (zero_point_mag - mag))
        
        x_pos = np.random.randint(10, simulator.image_size - 10)
        y_pos = np.random.randint(10, simulator.image_size - 10)
        
        size = 6
        star_kernel = np.zeros((2*size+1, 2*size+1), dtype=np.float32)
        star_kernel[size, size] = amplitude
        star_kernel = convolve(star_kernel, psf_kernel, boundary='extend')
        
        if add_spikes and mag < 12.0:
            spike_kernel = _create_spike_kernel(size=30, angle_offset=45)
            spike_intensity = amplitude * 0.05
            enhanced_kernel = star_kernel + spike_intensity * spike_kernel[:star_kernel.shape[0], :star_kernel.shape[1]]
            star_kernel = enhanced_kernel
        
        y_min, y_max = max(0, y_pos - size), min(simulator.image_size, y_pos + size + 1)
        x_min, x_max = max(0, x_pos - size), min(simulator.image_size, x_pos + size + 1)
        sy, sx = star_image[y_min:y_max, x_min:x_max].shape
        star_image[y_min:y_max, x_min:x_max] += star_kernel[:sy, :sx]
    
    return star_image

def _create_spike_kernel(size=30, num_spikes=4, angle_offset=45, spike_width=1.0):
    """
    Creates a kernel for diffraction spikes around bright stars.
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    y, x = np.mgrid[-center:center+1, -center:center+1]
    
    for i in range(num_spikes):
        angle = np.deg2rad(i * (180.0 / (num_spikes/2)) + angle_offset)
        dist_from_line = np.abs(x * np.cos(angle) + y * np.sin(angle))
        spike = np.exp(-(dist_from_line**2) / (2 * spike_width**2))
        kernel += spike
    
    return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel