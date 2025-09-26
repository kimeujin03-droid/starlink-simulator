import numpy as np
from skimage.draw import line_aa
from astropy.time import Time, TimeDelta
from scipy.ndimage import gaussian_filter
from skyfield.api import EarthSatellite, load
import random

class SatelliteStreak:
    """
    A specialized class for generating and managing satellite streaks.
    It utilizes WCS (World Coordinate System) and observer information from
    LSSTAdvancedSimulator to calculate and draw realistic satellite trajectories.
    """
    def __init__(self, simulator):
        self.sim = simulator
        self.ts = load.timescale()

    def find_optimal_observation_time(self, tle_line1, tle_line2, search_days=365, max_attempts=1000):
        """
        Finds an 'optimal time' within a given period when a satellite (from TLE data)
        passes through the observation field of view.
        """
        satellite = EarthSatellite(tle_line1, tle_line2, 'SAT', self.ts)
        now = Time.now() - TimeDelta(np.random.uniform(0, 365*24*3600), format='sec')
        found_times = []

        for _ in range(max_attempts):
            random_hours = np.random.uniform(0, search_days * 24)
            test_time = now + TimeDelta(random_hours * 3600, format='sec')
            exposure_duration = TimeDelta(30, format='sec')

            times = self.ts.linspace(self.ts.from_astropy(test_time),
                                     self.ts.from_astropy(test_time + exposure_duration),
                                     100)

            geocentric = satellite.at(times)
            observer_at_time = self.sim.observer.at(times)
            topocentric = geocentric - observer_at_time
            ra, dec, _ = topocentric.radec()

            try:
                px, py = self.sim.wcs.world_to_pixel_values(ra.degrees, dec.degrees) # type: ignore
                valid = (px >= -50) & (px < self.sim.image_size + 50) & \
                        (py >= -50) & (py < self.sim.image_size + 50)

                if np.sum(valid) >= 20:
                    found_times.append(test_time)
            except Exception:
                continue

        if found_times:
            selected_time = random.choice(found_times)
            print(f"   ✅ Randomly selected from {len(found_times)} valid times: {selected_time.iso}")
            return selected_time
        else:
            print(f"   ❌ No suitable time found after {max_attempts} attempts.")
            return None

def add_tle_streak_component(simulator, image, tle_data, brightness=60000, width=2.5):
    """
    Adds satellite streaks to the image based on a list of TLE data.
    """
    final_image = image.copy()
    satellite_added = False
    
    for tle_pair in tle_data:
        try:
            optimal_time = simulator.streak_generator.find_optimal_observation_time(tle_pair[0], tle_pair[1])
            if optimal_time is None:
                print("   Failed to generate TLE-based streak. No streak added.")
                continue

            ts = simulator.streak_generator.ts
            satellite = EarthSatellite(tle_pair[0], tle_pair[1], 'SAT', ts)
            exposure_duration = TimeDelta(30, format='sec')
            num_steps = int(exposure_duration.sec * 20)
            times = ts.linspace(ts.from_astropy(optimal_time),
                               ts.from_astropy(optimal_time + exposure_duration),
                               num_steps)

            geocentric = satellite.at(times)
            observer_at_time = simulator.observer.at(times)
            topocentric = geocentric - observer_at_time
            ra, dec, distance = topocentric.radec()

            pixel_coords = simulator.wcs.world_to_pixel_values(ra.degrees, dec.degrees) # type: ignore
            px, py = pixel_coords[0], pixel_coords[1]

            valid = (px >= 0) & (px < simulator.image_size) & (py >= 0) & (py < simulator.image_size)

            if not np.any(valid):
                print("   Satellite not in field of view. No streak added.")
                continue

            streak_layer = np.zeros_like(final_image, dtype=np.float32)
            
            valid_distances = distance.km[valid]
            distance_factor = np.clip(1500.0 / np.mean(valid_distances), 0.3, 5.0) if len(valid_distances) > 0 else 1.0
            adjusted_brightness = brightness * distance_factor

            valid_px = px[valid]
            valid_py = py[valid]

            for i in range(len(valid_px) - 1):
                rr, cc, val = line_aa(int(valid_py[i]), int(valid_px[i]),
                                     int(valid_py[i+1]), int(valid_px[i+1]))
                mask = (rr >= 0) & (rr < simulator.image_size) & (cc >= 0) & (cc < simulator.image_size)
                streak_layer[rr[mask], cc[mask]] = np.maximum(streak_layer[rr[mask], cc[mask]],
                                                             val[mask] * adjusted_brightness)

            blurred_streak = gaussian_filter(streak_layer, sigma=width/2.355)
            final_image += blurred_streak
            satellite_added = True
            print(f"   ✅ TLE-based streak added ({np.sum(valid)} points, brightness: {adjusted_brightness:.0f})")

        except Exception as e:
            print(f"   ⚠️ Error during TLE streak calculation: {e}")
            continue
            
    simulator.simulation_metadata['satellite_added'] = satellite_added
    return final_image

def add_tle_streak_DEBUG_MODE(simulator, image):
    """
    Adds a simple diagonal streak for debugging purposes, without complex calculations.
    """
    streak_layer = np.zeros_like(image, dtype=np.float32)
    start_x, start_y = np.random.randint(0, simulator.image_size, 2)
    end_x, end_y = np.random.randint(0, simulator.image_size, 2)
    rr, cc, val = line_aa(start_y, start_x, end_y, end_x)
    mask = (rr >= 0) & (rr < simulator.image_size) & (cc >= 0) & (cc < simulator.image_size)
    streak_layer[rr[mask], cc[mask]] = val[mask] * 60000
    blurred_streak = gaussian_filter(streak_layer, sigma=1.5)
    print("   ✅ DEBUG streak added.")
    return image + blurred_streak

def add_random_streak_component(simulator, image, base_brightness=50000, base_width=2.5):
    """
    Adds a physically plausible random streak to the image without TLE calculations.

    This function simulates a satellite on a linear 3D path and projects it onto
    the 2D image plane, adjusting brightness and width based on distance and velocity.
    """
    streak_layer = np.zeros_like(image, dtype=np.float32)

    # 1. Define a random but plausible 3D satellite path
    # Assume a typical LEO altitude range (e.g., 400-700 km)
    altitude_km = np.random.uniform(400, 700)
    # Orbital velocity at this altitude (approximate)
    velocity_kms = np.sqrt(398600 / (6371 + altitude_km)) # ~7.5 km/s

    # Create a random 3D trajectory that crosses the field of view
    # The field of view size at the satellite's altitude
    fov_at_alt_km = 2 * altitude_km * np.tan(np.deg2rad(simulator.pixel_scale * simulator.image_size / 3600 / 2))
    
    # Define start and end points in 3D space (relative to observer)
    # Start from one side of the FoV, end on the other
    start_vec = np.random.randn(3)
    start_vec[2] = 0 # Start on the xy-plane relative to the line of sight
    start_vec = start_vec / np.linalg.norm(start_vec) * fov_at_alt_km * 1.5
    end_vec = -start_vec

    # Add depth (z-axis, line of sight)
    start_vec[2] = altitude_km
    end_vec[2] = altitude_km + np.random.uniform(-fov_at_alt_km*0.1, fov_at_alt_km*0.1) # slight change in altitude

    # 2. Project the 3D path onto the 2D image plane
    num_steps = 200
    path_3d = np.linspace(start_vec, end_vec, num_steps)
    
    # Simple perspective projection
    # Convert 3D path (km) to angular separation (degrees)
    ra_offset = np.rad2deg(path_3d[:, 0] / path_3d[:, 2])
    dec_offset = np.rad2deg(path_3d[:, 1] / path_3d[:, 2])

    # Get center RA/Dec from WCS
    center_ra, center_dec = simulator.wcs.wcs.crval

    # Project to pixel coordinates
    px, py = simulator.wcs.world_to_pixel_values(center_ra + ra_offset, center_dec + dec_offset)

    # 3. Calculate physical properties at each point
    distances_km = np.linalg.norm(path_3d, axis=1)
    
    # Angular velocity (pixels per second)
    pixel_dist = np.sqrt(np.diff(px)**2 + np.diff(py)**2)
    time_per_step = np.linalg.norm(end_vec - start_vec) / velocity_kms / num_steps
    angular_velocity_pps = pixel_dist / time_per_step

    # 4. Draw the streak with varying brightness and width
    for i in range(num_steps - 1):
        # Check if the segment is inside the image
        if not (0 <= px[i] < simulator.image_size and 0 <= py[i] < simulator.image_size):
            continue

        # --- Physics-based adjustments ---
        # a) Brightness depends on distance (inverse square law)
        # Using a reference distance of 550km
        distance_factor = (550.0 / distances_km[i])**2
        
        # b) Brightness depends on angular velocity (slower = brighter)
        # A satellite moving slower across the frame deposits more light per pixel.
        # Using a reference velocity of 50 pixels/sec
        velocity_factor = np.clip(50.0 / angular_velocity_pps[i], 0.5, 2.0)

        adjusted_brightness = base_brightness * distance_factor * velocity_factor

        # c) Width can also depend on distance/brightness (optional)
        adjusted_width = base_width * np.sqrt(distance_factor)

        # Draw the line segment
        rr, cc, val = line_aa(int(py[i]), int(px[i]), int(py[i+1]), int(px[i+1]))
        mask = (rr >= 0) & (rr < simulator.image_size) & (cc >= 0) & (cc < simulator.image_size)
        
        # Apply blur for this segment to simulate width
        segment_layer = np.zeros_like(image, dtype=np.float32)
        segment_layer[rr[mask], cc[mask]] = val[mask] * adjusted_brightness
        
        # A small sigma for blurring each segment
        if np.sum(segment_layer) > 0:
            streak_layer += gaussian_filter(segment_layer, sigma=adjusted_width / 2.355)

    print("   ✅ Physically plausible random streak added.")
    simulator.simulation_metadata['satellite_added'] = True
    return image + streak_layer