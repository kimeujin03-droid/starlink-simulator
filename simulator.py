import numpy as np
import random
from astropy.wcs import WCS
from skyfield.api import Topos

# Import the newly created modules
from sky_objects import generate_galaxy_component, add_stars_component
from instrument_effects import apply_psf, add_ccd_noise, add_lsst_cosmic_rays, add_blooming
from satellite_streaks import SatelliteStreak, add_tle_streak_component

class LSSTAdvancedSimulator:
    """
    An advanced image simulator for the LSST/Vera C. Rubin Observatory,
    capable of generating realistic astronomical images including satellite streaks.
    """
    
    def __init__(self, image_size=512, seed=42):
        """
        Initializes the simulator with various LSST parameters and a random seed.
        
        Args:
            image_size (int): The size of the square image in pixels.
            seed (int): The random seed for reproducibility.
        """
        self.image_size = image_size
        self.pixel_scale = 0.2  # LSST pixel scale: 0.2 arcsec/pixel
        self.primary_diameter = 6.423  # Primary mirror effective diameter (m)
        self.field_of_view = 9.6  # Field of view in square degrees
        
        # LSST system parameters (simplified for simulation)
        self.standard_exposure = 15.0  # Standard visit exposure time (seconds)
        self.read_noise_range = (5.4, 6.2)  # Read noise range (electrons)
        
        # Create a grid for image generation.
        self.y, self.x = np.mgrid[:image_size, :image_size].astype(np.float32)
        np.random.seed(seed)
        random.seed(seed)
        
        # LSST sky brightness (magnitudes per square arcsecond)
        self.sky_magnitudes = {
            'u': 22.9, 'g': 22.3, 'r': 21.2, 
            'i': 20.5, 'z': 19.6, 'y': 18.6
        }
        
        # Zero points by filter (for 30s exposure)
        self.zero_points_30s = {
            'u': 27.0, 'g': 28.3, 'r': 28.1, 
            'i': 27.9, 'z': 27.4, 'y': 26.5
        }
        
        # Initialize WCS (World Coordinate System) information.
        self.wcs = self._create_default_wcs()
        
        # LSST observatory location (Cerro Pachon, Chile)
        self.observer = Topos('30.2444 S', '70.7494 W', elevation_m=2663)

        # Instantiate the SatelliteStreak generator.
        self.streak_generator = SatelliteStreak(self)
        
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        print(f"LSST Advanced Simulator initialized (image size: {image_size}x{image_size})")

    def _create_default_wcs(self):
        """
        Creates default WCS (World Coordinate System) information for a random sky area.
        WCS maps pixel coordinates to real-world celestial coordinates (RA, Dec).
        """
        wcs = WCS(naxis=2)
        
        # Set a random sky area, typically near the equator for higher satellite visibility.
        ra_center = np.random.uniform(0, 360)  # Right Ascension in degrees
        dec_center = np.random.uniform(-30, 30)  # Declination in degrees
        
        # WCS parameters configuration.
        wcs.wcs.crpix = [self.image_size/2, self.image_size/2]  # Reference pixel (image center)
        wcs.wcs.crval = [ra_center, dec_center]  # Celestial coordinates at the reference pixel
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"] # Coordinate system type
        # Pixel scale conversion from arcsec/pixel to degrees/pixel.
        wcs.wcs.cdelt = [-self.pixel_scale/3600, self.pixel_scale/3600]
        
        return wcs

    def generate_image(self, filter_band='r', exposure_time=30, include_cosmic_rays=True, 
                      include_blooming=True, include_satellites=True, satellite_probability=0.8, 
                      tle_data=None, verbose=True):
        """
        Generates a single LSST observation image by combining all simulated components.
        This is the main function orchestrating the image generation process.

        Args:
            filter_band (str): The filter band ('u', 'g', 'r', 'i', 'z', 'y').
            exposure_time (int): The exposure time in seconds.
            include_cosmic_rays (bool): Whether to include cosmic ray effects.
            include_blooming (bool): Whether to include CCD blooming effects.
            include_satellites (bool): Whether to include satellite streaks.
            satellite_probability (float): Probability (0-1) of a satellite streak appearing.
            tle_data (list of tuples): List of TLE data tuples [(line1, line2), ...].
            verbose (bool): Whether to print detailed status messages.

        Returns:
            ndarray: The final simulated image.
        """
        if verbose:
            print(f"\nLSST {filter_band}-band observation simulation started (exposure: {exposure_time}s)")

        # Reset metadata for this run
        self.simulation_metadata = {}

        # [Step 1] Generate ideal sky objects (galaxy and stars).
        ideal_image = generate_galaxy_component(self, filter_band)
        ideal_image = add_stars_component(self, ideal_image, galactic_latitude=5.0, filter_band=filter_band)

        # [Step 3] Apply atmospheric and telescope effects (PSF).
        convolved_image = apply_psf(self, ideal_image)

        # [Step 4] Add CCD noise components.
        noisy_image = add_ccd_noise(self, convolved_image, filter_band, exposure_time)
        
        # [Step 5] Add cosmic rays.
        final_image = noisy_image.copy()
        if include_cosmic_rays:
            final_image = add_lsst_cosmic_rays(self, noisy_image, exposure_time)

        # [Step 6] Add satellite streaks.
        if include_satellites and np.random.random() < satellite_probability:
            if tle_data:
                final_image = add_tle_streak_component(self, final_image, tle_data)
        
        # [Step 7] Add blooming effect.
        if include_blooming:
            final_image = add_blooming(self, final_image)

        galaxy_type = self.simulation_metadata.get('galaxy_type', 'unknown')
        cr_count = self.simulation_metadata.get('cosmic_ray_count', 0)
        satellite_added = self.simulation_metadata.get('satellite_added', False)
        
        if verbose:
            status_msg = f"Simulation complete! ({galaxy_type} galaxy, {cr_count} cosmic rays"
            if satellite_added:
                status_msg += ", satellite streak included"
            status_msg += ")"
            print(status_msg)

        return final_image

# --- Sample TLE Data for Starlink Satellites ---
SAMPLE_TLE_DATA = {
    'STARLINK-G4 (53.2 deg)': ( # Starlink satellite with 53.2 deg inclination, high visibility for LSST.
        "1 53099U 22082CH  24155.50000000  .00002100  00000+0  42000-3 0  9991",
        "2 53099  53.2173 211.2053 0001500 105.3000 254.8000 15.08200000 98001"
    ),
    'STARLINK-G3 (69.9 deg)': ( # Starlink satellite with 69.9 deg inclination, better for Southern Hemisphere.
        "1 56814U 23091K   24155.50000000  .00002500  00000+0  48000-3 0  9992",
        "2 56814  69.9980 180.0000 0001200 135.0000 225.1000 15.11500000 45008"
    )
}