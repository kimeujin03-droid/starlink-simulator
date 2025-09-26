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