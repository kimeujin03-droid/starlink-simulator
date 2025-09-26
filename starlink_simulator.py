import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, Moffat2DKernel
from scipy.ndimage import gaussian_filter
from skyfield.api import EarthSatellite, load, Topos

class SatelliteStreak:
    """
    A specialized class for generating and managing satellite streaks.
    It utilizes WCS (World Coordinate System) and observer information from
    LSSTAdvancedSimulator to calculate and draw realistic satellite trajectories.
    """
    def __init__(self, simulator):
        # simulator: An instance of LSSTAdvancedSimulator, providing access to image size, WCS, etc.
        self.sim = simulator
        # ts: skyfield's timescale object, essential for astronomical time calculations.
        self.ts = load.timescale()

    def find_optimal_observation_time(self, tle_line1, tle_line2, search_days=365, max_attempts=1000, return_all=False):
        """
        Finds an 'optimal time' within a given period when a satellite (from TLE data)
        passes through the observation field of view.

        Args:
            tle_line1 (str): The first line of TLE data.
            tle_line2 (str): The second line of TLE data.
            search_days (int): The period (in days) to search. Default is 365 days.
            max_attempts (int): Number of random attempts to find an optimal time.
            return_all (bool): If True, returns a list of all valid times found.
                               If False, returns one randomly selected time.

        Returns:
            astropy.time.Time or list or None: The found observation time(s), or None if not found.
        """
        # Create a skyfield satellite object from TLE data.
        satellite = EarthSatellite(tle_line1, tle_line2, 'SAT', self.ts)
        # Set a random starting point for the search (within the last year)
        now = Time.now() - TimeDelta(np.random.uniform(0, 365*24*3600), format='sec')
        # List to store valid times found.
        found_times = []

        # Iterate for a specified number of attempts.
        for _ in range(max_attempts):
            # Select a random time within the search period.
            random_hours = np.random.uniform(0, search_days * 24)
            test_time = now + TimeDelta(random_hours * 3600, format='sec')
            exposure_duration = TimeDelta(30, format='sec')

            # Calculate the satellite's position at 100 points during the 30-second exposure.
            times = self.ts.linspace(self.ts.from_astropy(test_time),
                                     self.ts.from_astropy(test_time + exposure_duration),
                                     100)

            # 1. Calculate satellite's geocentric position.
            geocentric = satellite.at(times)
            # 2. Calculate LSST observatory's position.
            observer_at_time = self.sim.observer.at(times)
            # 3. Calculate satellite's topocentric position (as seen from the observatory).
            topocentric = geocentric - observer_at_time
            # 4. Convert to celestial coordinates (Right Ascension, Declination).
            ra, dec, _ = topocentric.radec()

            try:
                # Convert celestial coordinates to image pixel coordinates (px, py).
                px, py = self.sim.wcs.world_to_pixel_values(ra.degrees, dec.degrees) # type: ignore
                # Check if pixel coordinates fall within the image area (+ a small margin).
                valid = (px >= -50) & (px < self.sim.image_size + 50) & \
                        (py >= -50) & (py < self.sim.image_size + 50)

                # If at least 20 points of the trajectory are within the field of view, consider it valid.
                if np.sum(valid) >= 20:
                    # If not returning all times, return the first valid time found immediately.
                    if not return_all:
                        print(f"   ✅ First valid observation time found: {test_time.iso}")
                        return test_time
                    # If returning all times, add the found time to the list.
                    found_times.append(test_time)
            except Exception:
                # Ignore errors during coordinate conversion and continue to the next attempt.
                continue

        # After all attempts, process the found times.
        if return_all:
            # If return_all is True, return the entire list of found times.
            print(f"   🔍 Found {len(found_times)} valid times.")
            return found_times
        elif found_times:
            # If times were found, select one randomly and return it.
            selected_time = random.choice(found_times)
            print(f"   ✅ Randomly selected from {len(found_times)} valid times: {selected_time.iso}")
            return selected_time
        else:
            # If no valid time was found after all attempts, return None.
            print(f"   ❌ No suitable time found after {max_attempts} attempts.")
            return None

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

    def _generate_realistic_galaxy_params(self, filter_band='r'):
        """
        Generates realistic galaxy parameters (e.g., brightness, size, shape)
        based on observed astronomical distributions.
        """
        # 1. Determine galaxy magnitude (brightness) randomly and convert to amplitude.
        zero_point_mag = self.zero_points_30s.get(filter_band, 28.1) 
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
            'r_eff': np.clip(r_eff_pixels, 1.0, self.image_size // 6),
            'theta': np.random.uniform(0, np.pi), # Random orientation
            'x_0': self.image_size / 2 + np.random.normal(0, 3.0), # Slightly offset from center
            'y_0': self.image_size / 2 + np.random.normal(0, 3.0)
        })
        
        return params

    def _get_sky_background_counts(self, filter_band, exposure_time):
        """
        Calculates the sky background brightness in electron counts per pixel.
        The night sky is not perfectly dark and contributes to image noise.
        """
        sky_mag = self.sky_magnitudes.get(filter_band, 21.2)
        # Factor for light collection based on primary mirror diameter.
        collecting_area_factor = (self.primary_diameter / 8.4) ** 2
        sky_counts_per_sec = 10**((25 - sky_mag) / 2.5) * (self.pixel_scale**2) * collecting_area_factor
        
        if exposure_time < 0:
            raise ValueError("exposure_time must be non-negative")
        return sky_counts_per_sec * exposure_time

    def _add_lsst_cosmic_rays(self, image, exposure_time):
        """
        Adds cosmic ray effects to the image, simulating high-energy particles
        hitting the CCD sensor.
        """
        final_image = image.copy()
        
        # Estimate expected number of cosmic rays based on LSST rates and exposure time.
        cr_rate_per_15s = np.random.uniform(2.0, 3.0) 
        detector_area_fraction = (self.image_size / 4096) ** 2 # Scale for image size
        expected_crs = cr_rate_per_15s * (exposure_time / 15.0) * detector_area_fraction
        expected_crs = max(1, min(expected_crs, 50)) # Clip to a reasonable range
        
        # Use Poisson distribution to determine the actual number of cosmic rays.
        num_cosmic_rays = np.random.poisson(expected_crs)
        
        # Define probabilities for different cosmic ray morphologies (track, spot, worm).
        morphology_weights = ['track'] * 7 + ['spot'] * 2 + ['worm'] * 1
        
        # Add each cosmic ray with random energy and morphology.
        for _ in range(num_cosmic_rays):
            cr_energy = np.clip(np.random.lognormal(np.log(20000), 0.7), 2000, 80000)
            morphology = random.choice(morphology_weights)
            
            margin = 15 # Ensure cosmic rays are not too close to the image edge.
            start_x = np.random.randint(margin, self.image_size - margin)
            start_y = np.random.randint(margin, self.image_size - margin)
            
            if morphology == 'spot':
                self._add_cosmic_ray_spot(final_image, start_x, start_y, cr_energy)
            elif morphology == 'worm':
                self._add_cosmic_ray_worm(final_image, start_x, start_y, cr_energy)
            else:  # 'track'
                self._add_cosmic_ray_track(final_image, start_x, start_y, cr_energy)
        
        return final_image, num_cosmic_rays

    def _add_cosmic_ray_spot(self, image, x, y, energy):
        """Adds a point-like cosmic ray, modeled as a Gaussian blob."""
        spot_size = np.random.uniform(0.8, 1.2)
        y_grid, x_grid = np.mgrid[-2:3, -2:3]
        spot_kernel = np.exp(-((x_grid**2 + y_grid**2) / (2 * spot_size**2)))
        
        y_slice = slice(max(0, y-2), min(self.image_size, y+3))
        x_slice = slice(max(0, x-2), min(self.image_size, x+3))
        sy, sx = image[y_slice, x_slice].shape
        image[y_slice, x_slice] += spot_kernel[:sy, :sx] * energy

    def _add_cosmic_ray_track(self, image, start_x, start_y, energy):
        """Adds a linear track cosmic ray."""
        track_length = int(np.clip(np.random.exponential(12.0), 4, 40))
        angle = np.random.uniform(0, 2 * np.pi)
        
        for step in range(track_length):
            x_pos = int(start_x + step * np.cos(angle))
            y_pos = int(start_y + step * np.sin(angle))
            
            if 0 <= x_pos < self.image_size and 0 <= y_pos < self.image_size:
                # Energy fraction decreases along the track.
                energy_fraction = np.exp(-step / (track_length * 0.7)) if track_length > 0 else 1
                image[y_pos, x_pos] += energy * energy_fraction / (track_length * 0.5 + 1)

    def _add_cosmic_ray_worm(self, image, start_x, start_y, energy):
        """Adds a worm-like (curved) cosmic ray."""
        track_length = int(np.clip(np.random.exponential(15.0), 6, 35))
        angle = np.random.uniform(0, 2 * np.pi)
        waviness = np.random.uniform(1.5, 3.5) # How much it wiggles
        frequency = np.random.uniform(0.4, 0.7) # How often it wiggles
        
        for step in range(track_length):
            # Base linear path.
            base_x = start_x + step * np.cos(angle)
            base_y = start_y + step * np.sin(angle)
            
            # Add sinusoidal offset for waviness.
            offset_x = waviness * np.sin(frequency * step) * (-np.sin(angle))
            offset_y = waviness * np.sin(frequency * step) * np.cos(angle)
            
            x_pos = int(base_x + offset_x)
            y_pos = int(base_y + offset_y)
            
            if 0 <= x_pos < self.image_size and 0 <= y_pos < self.image_size:
                image[y_pos, x_pos] += energy / (track_length * 0.8)

    def _add_stars(self, image, galactic_latitude=30.0, filter_band='r', add_spikes=True):
        """
        Adds background stars to the image, simulating stellar density based on
        galactic latitude and applying PSF and diffraction spikes for bright stars.
        """
        star_image = image.copy()
        
        # Simulate higher star density closer to the galactic plane.
        lat_factor = 1.0 / (np.abs(np.sin(np.deg2rad(galactic_latitude))) + 0.1)
        base_density = 10000 # Base star density
        star_density = base_density * min(lat_factor, 5.0)
        patch_area_deg2 = (self.image_size * self.pixel_scale / 3600)**2
        expected_stars = star_density * patch_area_deg2
        num_stars = np.random.poisson(expected_stars)
        num_stars = min(num_stars, 200) # Limit maximum number of stars for performance.
        
        # Create PSF (Point Spread Function) kernel to simulate atmospheric seeing.
        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.15), 0.4, 1.2)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.8)
        
        for _ in range(num_stars):
            # Randomly determine star magnitude (brightness), favoring dimmer stars.
            zero_point_mag = 28.1
            mag = np.random.power(2.35) * 18 + 12 # Power-law distribution for magnitudes.
            amplitude = 10**(0.4 * (zero_point_mag - mag))
            
            # Randomly determine star position.
            x_pos = np.random.randint(10, self.image_size - 10)
            y_pos = np.random.randint(10, self.image_size - 10)
            
            # Create an ideal point source (star) and convolve with PSF.
            size = 6
            star_kernel = np.zeros((2*size+1, 2*size+1), dtype=np.float32)
            star_kernel[size, size] = amplitude
            star_kernel = convolve(star_kernel, psf_kernel, boundary='extend')
            
            # Add diffraction spikes for very bright stars (mag < 12.0).
            if add_spikes and mag < 12.0:
                spike_kernel = self._create_spike_kernel(size=30, angle_offset=45)
                spike_intensity = amplitude * 0.05
                enhanced_kernel = star_kernel + spike_intensity * spike_kernel[:star_kernel.shape[0], :star_kernel.shape[1]]
                star_kernel = enhanced_kernel
            
            # Add the generated star to the overall star image.
            y_min, y_max = max(0, y_pos - size), min(self.image_size, y_pos + size + 1)
            x_min, x_max = max(0, x_pos - size), min(self.image_size, x_pos + size + 1)
            sy, sx = star_image[y_min:y_max, x_min:x_max].shape
            star_image[y_min:y_max, x_min:x_max] += star_kernel[:sy, :sx]
        
        return star_image

    def _create_spike_kernel(self, size=30, num_spikes=4, angle_offset=45, spike_width=1.0):
        """
        Creates a kernel for diffraction spikes, which appear around bright stars
        due to the telescope's internal structure.
        """
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        y, x = np.mgrid[-center:center+1, -center:center+1]
        
        for i in range(num_spikes):
            angle = np.deg2rad(i * (180.0 / (num_spikes/2)) + angle_offset)
            # Calculate distance from the line representing the spike.
            dist_from_line = np.abs(x * np.cos(angle) + y * np.sin(angle))
            spike = np.exp(-(dist_from_line**2) / (2 * spike_width**2))
            kernel += spike
        
        # Normalize the kernel.
        return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

    def _add_blooming(self, image, saturation_limit=65000, bleed_fraction=0.1, decay_factor=0.5):
        """
        Simulates CCD blooming effect, where excess charge from saturated pixels
        bleeds into adjacent pixels, typically vertically.
        """
        bloomed_image = image.copy()
        # Find coordinates of saturated pixels.
        saturated_coords = np.argwhere(bloomed_image > saturation_limit)
        
        if saturated_coords.shape[0] == 0:
            return bloomed_image
        
        # Process blooming column by column.
        for c in np.unique(saturated_coords[:, 1]):
            col_saturated_rows = sorted(saturated_coords[saturated_coords[:, 1] == c][:, 0])
            
            for r_start in col_saturated_rows:
                # Calculate excess charge beyond saturation limit.
                excess_charge = (bloomed_image[r_start, c] - saturation_limit) * bleed_fraction
                bloomed_image[r_start, c] = saturation_limit # Cap pixel value at saturation.
                
                # Bleed excess charge upwards and downwards.
                for direction in [-1, 1]:
                    charge_to_bleed = excess_charge / 2.0 # Split charge for two directions.
                    for step in range(1, self.image_size):
                        r = r_start + direction * step
                        # Stop bleeding if out of bounds or charge is negligible.
                        if not (0 <= r < self.image_size) or charge_to_bleed < 1:
                            break
                        
                        bloomed_image[r, c] += charge_to_bleed
                        # Charge decays with distance.
                        charge_to_bleed *= decay_factor
        
        return bloomed_image

    def _add_tle_streak(self, image, tle_line1, tle_line2, brightness=60000, width=2.5, optimal_time=None):
        """
        Adds a satellite streak to the image based on TLE data.

        Args:
            image (ndarray): The original image to add the streak to.
            tle_line1, tle_line2 (str): TLE data for the satellite.
            brightness (float): Base brightness of the streak.
            width (float): Thickness of the streak in pixels.
            optimal_time (astropy.time.Time, optional): Specific time to draw the streak.
                                                        If None, it will be automatically searched.

        Returns:
            tuple: (Image with streak, boolean indicating success)
        """
        try:
            # If optimal_time is not provided, automatically search for one.
            if optimal_time is None:
                optimal_time = self.streak_generator.find_optimal_observation_time(tle_line1, tle_line2)
            if optimal_time is None:
                print("   Failed to generate TLE-based streak. No streak added.")
                return image, False

            # Calculate satellite trajectory for the given exposure duration.
            ts = self.streak_generator.ts
            satellite = EarthSatellite(tle_line1, tle_line2, 'SAT', ts)
            exposure_duration = TimeDelta(30, format='sec')
            num_steps = int(exposure_duration.sec * 20) # High resolution for trajectory.
            times = ts.linspace(ts.from_astropy(optimal_time),
                               ts.from_astropy(optimal_time + exposure_duration),
                               num_steps)

            geocentric = satellite.at(times)
            observer_at_time = self.observer.at(times)
            topocentric = geocentric - observer_at_time
            ra, dec, distance = topocentric.radec()

            # Convert calculated trajectory to image pixel coordinates.
            pixel_coords = self.wcs.world_to_pixel_values(ra.degrees, dec.degrees) # type: ignore
            px, py = pixel_coords[0], pixel_coords[1]

            # Filter for trajectory points that are within the image boundaries.
            valid = (px >= 0) & (px < self.image_size) & (py >= 0) & (py < self.image_size)

            if not np.any(valid):
                print("   Satellite not in field of view. No streak added.")
                return image, False

            # Create an empty layer for the streak.
            streak_layer = np.zeros_like(image, dtype=np.float32)
            
            # Adjust streak brightness based on satellite distance (closer = brighter).
            valid_distances = distance.km[valid]
            distance_factor = np.clip(1500.0 / np.mean(valid_distances), 0.3, 5.0) if len(valid_distances) > 0 else 1.0
            adjusted_brightness = brightness * distance_factor

            valid_px = px[valid]
            valid_py = py[valid]

            # Draw the streak by connecting trajectory points with anti-aliased lines.
            for i in range(len(valid_px) - 1):
                rr, cc, val = line_aa(int(valid_py[i]), int(valid_px[i]),
                                     int(valid_py[i+1]), int(valid_px[i+1]))
                # Ensure drawn lines are within image bounds.
                mask = (rr >= 0) & (rr < self.image_size) & (cc >= 0) & (cc < self.image_size)
                streak_layer[rr[mask], cc[mask]] = np.maximum(streak_layer[rr[mask], cc[mask]],
                                                             val[mask] * adjusted_brightness)

            # Apply Gaussian blur to simulate streak width.
            blurred_streak = gaussian_filter(streak_layer, sigma=width/2.355)
            print(f"   ✅ TLE-based streak added ({np.sum(valid)} points, brightness: {adjusted_brightness:.0f})")
            return image + blurred_streak, True

        except Exception as e:
            print(f"   ⚠️ Error during TLE streak calculation: {e}")
            return image, False

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

        # [Step 1] Generate an ideal galaxy using Sersic profile.
        galaxy_params = self._generate_realistic_galaxy_params(filter_band)
        allowed_keys = ["amplitude", "r_eff", "n", "x_0", "y_0", "ellip", "theta"]
        sersic_params = {k: v for k, v in galaxy_params.items() if k in allowed_keys}
        ideal_image = Sersic2D(**sersic_params)(self.x, self.y)

        # [Step 2] Add background stars.
        ideal_image = self._add_stars(ideal_image, galactic_latitude=5.0, filter_band=filter_band, add_spikes=True)

        # [Step 3] Apply atmospheric and telescope effects (PSF).
        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.2), 0.4, 1.5)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.5)
        convolved_image = convolve(ideal_image, psf_kernel, boundary='extend')

        # [Step 4] Add CCD noise components.
        # 4-1. Sky background noise.
        sky_counts = self._get_sky_background_counts(filter_band, exposure_time)
        # 4-2. Dark current noise (from thermal generation in CCD).
        dark_counts = 0.002 * exposure_time
        
        base_signal = convolved_image + sky_counts + dark_counts
        # 4-3. Shot noise (Poisson noise from photon counting statistics).
        image_with_shot_noise = np.random.poisson(np.maximum(base_signal, 0))
        
        # 4-4. Read noise (from reading out the CCD sensor).
        read_noise = np.random.normal(0, 8.0, self.image_size**2).reshape(self.image_size, self.image_size)
        noisy_image = image_with_shot_noise + read_noise
        
        # [Step 5] Add cosmic rays.
        final_image = noisy_image.copy()
        cr_count = 0
        if include_cosmic_rays:
            final_image, cr_count = self._add_lsst_cosmic_rays(noisy_image, exposure_time)
        
        # [Step 6] Add satellite streaks.
        satellite_added = False
        if include_satellites and np.random.random() < satellite_probability:
            if tle_data:
                # Iterate through all provided TLE data (for multiple streaks if needed).
                for tle_pair in tle_data:
                    final_image, added = self._add_tle_streak(
                        final_image, tle_pair[0], tle_pair[1], optimal_time=None) # No intersection_time for single streak
                    satellite_added = satellite_added or added
        
        # [Step 7] Add blooming effect.
        if include_blooming:
            final_image = self._add_blooming(final_image)

        galaxy_type = galaxy_params.get('type', 'unknown')
        
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

def main():
    """
    Main function to demonstrate the LSSTAdvancedSimulator.
    Generates and displays a single realistic astronomical image with a Starlink streak.
    """
    print("🚀 Starting LSST Advanced Simulator to generate an image...")

    # 1. Initialize the simulator.
    # Change the 'seed' value for different galaxy, star, and noise patterns.
    simulator = LSSTAdvancedSimulator(image_size=512, seed=2024)

    # 2. Select Starlink TLE data for simulation.
    # Choose one from the SAMPLE_TLE_DATA defined above.
    tle_data = SAMPLE_TLE_DATA['STARLINK-G4 (53.2 deg)']
    
    # 3. Generate the final image including a satellite streak.
    # The generate_image function handles all complex internal processes.
    sim_image = simulator.generate_image(
        filter_band='r',
        exposure_time=30,
        include_satellites=True,
        satellite_probability=1.0, # 100% probability to ensure a streak is generated.
        tle_data=[tle_data]        # TLE data must be passed as a list.
    )

    # 4. Visualize the generated image.
    if sim_image is not None:
        plt.figure(figsize=(10, 10))
        # Adjust vmin and vmax for optimal contrast to reveal faint objects and streaks.
        vmin = np.percentile(sim_image, 1)
        vmax = np.percentile(sim_image, 99.8)
        
        plt.imshow(sim_image, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        plt.title('LSST Simulated Image with Starlink Streak', fontsize=16)
        plt.axis('off') # Hide axes for a cleaner astronomical image look.
        plt.tight_layout()
        plt.show()

# --- Main execution block ---
# This ensures the 'main()' function runs only when the script is executed directly.
if __name__ == '__main__':
    print("LSST Starlink Streak Simulator")
    print("=" * 50)
    main()
    print("\n✅ Simulation complete!")