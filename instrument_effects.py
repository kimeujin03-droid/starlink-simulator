import numpy as np
import random
from astropy.convolution import convolve, Moffat2DKernel

def apply_psf(simulator, image):
    """
    Applies atmospheric and telescope effects (PSF) to the image.
    """
    seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.2), 0.4, 1.5)
    psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / simulator.pixel_scale / 2.35, alpha=2.5)
    return convolve(image, psf_kernel, boundary='extend')

def get_sky_background_counts(simulator, filter_band, exposure_time):
    """
    Calculates the sky background brightness in electron counts per pixel.
    """
    sky_mag = simulator.sky_magnitudes.get(filter_band, 21.2)
    collecting_area_factor = (simulator.primary_diameter / 8.4) ** 2
    sky_counts_per_sec = 10**((25 - sky_mag) / 2.5) * (simulator.pixel_scale**2) * collecting_area_factor
    
    if exposure_time < 0:
        raise ValueError("exposure_time must be non-negative")
    return sky_counts_per_sec * exposure_time

def add_lsst_cosmic_rays(simulator, image, exposure_time):
    """
    Adds cosmic ray effects to the image.
    """
    final_image = image.copy()
    
    cr_rate_per_15s = np.random.uniform(2.0, 3.0) 
    detector_area_fraction = (simulator.image_size / 4096) ** 2
    expected_crs = cr_rate_per_15s * (exposure_time / 15.0) * detector_area_fraction
    expected_crs = max(1, min(expected_crs, 50))
    
    num_cosmic_rays = np.random.poisson(expected_crs)
    
    morphology_weights = ['track'] * 7 + ['spot'] * 2 + ['worm'] * 1
    
    for _ in range(num_cosmic_rays):
        cr_energy = np.clip(np.random.lognormal(np.log(20000), 0.7), 2000, 80000)
        morphology = random.choice(morphology_weights)
        
        margin = 15
        start_x = np.random.randint(margin, simulator.image_size - margin)
        start_y = np.random.randint(margin, simulator.image_size - margin)
        
        if morphology == 'spot':
            _add_cosmic_ray_spot(simulator, final_image, start_x, start_y, cr_energy)
        elif morphology == 'worm':
            _add_cosmic_ray_worm(simulator, final_image, start_x, start_y, cr_energy)
        else:
            _add_cosmic_ray_track(simulator, final_image, start_x, start_y, cr_energy)
    
    simulator.simulation_metadata['cosmic_ray_count'] = num_cosmic_rays
    return final_image

def _add_cosmic_ray_spot(simulator, image, x, y, energy):
    """Adds a point-like cosmic ray."""
    spot_size = np.random.uniform(0.8, 1.2)
    y_grid, x_grid = np.mgrid[-2:3, -2:3]
    spot_kernel = np.exp(-((x_grid**2 + y_grid**2) / (2 * spot_size**2)))
    
    y_slice = slice(max(0, y-2), min(simulator.image_size, y+3))
    x_slice = slice(max(0, x-2), min(simulator.image_size, x+3))
    sy, sx = image[y_slice, x_slice].shape
    image[y_slice, x_slice] += spot_kernel[:sy, :sx] * energy

def _add_cosmic_ray_track(simulator, image, start_x, start_y, energy):
    """Adds a linear track cosmic ray."""
    track_length = int(np.clip(np.random.exponential(12.0), 4, 40))
    angle = np.random.uniform(0, 2 * np.pi)
    
    for step in range(track_length):
        x_pos = int(start_x + step * np.cos(angle))
        y_pos = int(start_y + step * np.sin(angle))
        
        if 0 <= x_pos < simulator.image_size and 0 <= y_pos < simulator.image_size:
            energy_fraction = np.exp(-step / (track_length * 0.7)) if track_length > 0 else 1
            image[y_pos, x_pos] += energy * energy_fraction / (track_length * 0.5 + 1)

def _add_cosmic_ray_worm(simulator, image, start_x, start_y, energy):
    """Adds a worm-like (curved) cosmic ray."""
    track_length = int(np.clip(np.random.exponential(15.0), 6, 35))
    angle = np.random.uniform(0, 2 * np.pi)
    waviness = np.random.uniform(1.5, 3.5)
    frequency = np.random.uniform(0.4, 0.7)
    
    for step in range(track_length):
        base_x = start_x + step * np.cos(angle)
        base_y = start_y + step * np.sin(angle)
        
        offset_x = waviness * np.sin(frequency * step) * (-np.sin(angle))
        offset_y = waviness * np.sin(frequency * step) * np.cos(angle)
        
        x_pos = int(base_x + offset_x)
        y_pos = int(base_y + offset_y)
        
        if 0 <= x_pos < simulator.image_size and 0 <= y_pos < simulator.image_size:
            image[y_pos, x_pos] += energy / (track_length * 0.8)

def add_blooming(simulator, image, saturation_limit=65000, bleed_fraction=0.1, decay_factor=0.5):
    """
    Simulates CCD blooming effect.
    """
    bloomed_image = image.copy()
    saturated_coords = np.argwhere(bloomed_image > saturation_limit)
    
    if saturated_coords.shape[0] == 0:
        return bloomed_image
    
    for c in np.unique(saturated_coords[:, 1]):
        col_saturated_rows = sorted(saturated_coords[saturated_coords[:, 1] == c][:, 0])
        
        for r_start in col_saturated_rows:
            excess_charge = (bloomed_image[r_start, c] - saturation_limit) * bleed_fraction
            bloomed_image[r_start, c] = saturation_limit
            
            for direction in [-1, 1]:
                charge_to_bleed = excess_charge / 2.0
                for step in range(1, simulator.image_size):
                    r = r_start + direction * step
                    if not (0 <= r < simulator.image_size) or charge_to_bleed < 1:
                        break
                    
                    bloomed_image[r, c] += charge_to_bleed
                    charge_to_bleed *= decay_factor
    
    return bloomed_image

def add_ccd_noise(simulator, image, filter_band, exposure_time):
    """
    Adds various CCD noise components (sky background, dark current, shot noise, read noise).
    """
    # Sky background noise
    sky_counts = get_sky_background_counts(simulator, filter_band, exposure_time)
    # Dark current noise
    dark_counts = 0.002 * exposure_time
    
    base_signal = image + sky_counts + dark_counts
    # Shot noise (Poisson noise)
    image_with_shot_noise = np.random.poisson(np.maximum(base_signal, 0))
    
    # Read noise
    read_noise = np.random.normal(0, 8.0, simulator.image_size**2).reshape(simulator.image_size, simulator.image_size)
    noisy_image = image_with_shot_noise + read_noise
    
    return noisy_image