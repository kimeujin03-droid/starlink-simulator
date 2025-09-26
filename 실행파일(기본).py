import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, Moffat2DKernel
from scipy.ndimage import gaussian_filter

# Note: skimage and skyfield are not available in the environment, so parts using them are commented out or replaced with simple approximations.

class SatelliteStreak:
    """
    위성 스트릭(streak) 생성 및 관리 를 위한 전문 클래스입니다.
    LSSTAdvancedSimulator로부터 WCS(좌표계), 관측자 정보 등을 받아
    현실적인 위성 궤적을 계산하고 그리는 역할 을 담당합니다.
    """
    def __init__(self, simulator):
        self.sim = simulator
        # ts: skyfield 라이브러리의 시간 척도(timescale) 객체. 천문 계산에 필요한 표준 시간 시스템입니다.
        # self.ts = load.timescale() # Commented out due to missing skyfield

    def find_optimal_observation_time(self, tle_line1, tle_line2, search_days=365, max_attempts=1000, return_all=False):
        # Due to missing skyfield, return a dummy time
        return Time.now()

    def find_intersection_time(self, tle_pair1, tle_pair2, search_days=365, max_attempts=1000):
        # Due to missing skyfield, return a dummy time
        return Time.now()

class LSSTAdvancedSimulator:
    """
    LSST/Vera C. Rubin Observatory advanced image simulator with satellite streaks.
    """
    
    def __init__(self, image_size=512, seed=42):
        self.image_size = image_size
        self.pixel_scale = 0.2
        self.primary_diameter = 6.423
        self.field_of_view = 9.6
        
        self.standard_exposure = 15.0
        self.visits_per_observation = 2
        self.read_noise_range = (5.4, 6.2)
        self.gain_range = (1.5, 1.7)
        self.r_band_5sigma_depth = 24.7
        
        self.y, self.x = np.mgrid[:image_size, :image_size].astype(np.float32)
        np.random.seed(seed)
        random.seed(seed)
        
        self.sky_magnitudes = {
            'u': 22.9, 'g': 22.3, 'r': 21.2, 
            'i': 20.5, 'z': 19.6, 'y': 18.6
        }
        
        self.zero_points_30s = {
            'u': 27.0, 'g': 28.3, 'r': 28.1, 
            'i': 27.9, 'z': 27.4, 'y': 26.5
        }
        
        self.wcs = self._create_default_wcs()
        
        self.observer = None # Topos not available
        
        self.streak_generator = SatelliteStreak(self)
        
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        print(f"LSST Advanced Simulator initialized (image size: {image_size}x{image_size})")

    def _create_default_wcs(self):
        wcs = WCS(naxis=2)
        
        ra_center = np.random.uniform(0, 360)
        dec_center = np.random.uniform(-30, 30)
        
        wcs.wcs.crpix = [self.image_size/2, self.image_size/2]
        wcs.wcs.crval = [ra_center, dec_center]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cdelt = [-self.pixel_scale/3600, self.pixel_scale/3600]
        
        return wcs

    def set_field_center(self, ra, dec):
        self.wcs.wcs.crval = [ra, dec]
        print(f"Field center set: RA={ra:.2f}°, Dec={dec:.2f}°")

    def _generate_realistic_galaxy_params(self, filter_band='r'):
        zero_point_mag = self.zero_points_30s.get(filter_band, 28.1) 
        mag_r = np.random.uniform(18.0, 24.5)
        amplitude = 10**(0.4 * (zero_point_mag - mag_r))
        
        galaxy_type_prob = np.random.random()
        if galaxy_type_prob < 0.3:
            params = {
                'type': 'elliptical',
                'n': np.clip(np.random.normal(4.0, 0.5), 0.5, 8.0),
                'r_eff_kpc': np.random.lognormal(np.log(4.0), 0.3),
                'ellip': np.clip(np.random.beta(2, 2) * 0.8, 0.0, 0.9),
                'amplitude': amplitude * 1.2
            }
        elif galaxy_type_prob < 0.8:
            params = {
                'type': 'spiral',
                'n': np.clip(np.random.normal(1.0, 0.2), 0.5, 8.0),
                'r_eff_kpc': np.random.lognormal(np.log(4.5), 0.4),
                'ellip': np.clip(np.random.beta(1.5, 3) * 0.6, 0.0, 0.9),
                'amplitude': amplitude
            }
        else:
            params = {
                'type': 'irregular',
                'n': np.random.uniform(0.5, 2.0),
                'r_eff_kpc': np.random.lognormal(np.log(2.5), 0.6),
                'ellip': np.clip(np.random.beta(1, 1) * 0.7, 0.0, 0.9),
                'amplitude': amplitude * 0.8
            }
        
        redshift_factor = np.random.uniform(0.8, 1.5)
        r_eff_pixels = params['r_eff_kpc'] * 5.0 / redshift_factor
        
        params.update({
            'amplitude': np.clip(params['amplitude'], 100, 5e5),
            'r_eff': np.clip(r_eff_pixels, 1.0, self.image_size // 6),
            'theta': np.random.uniform(0, np.pi),
            'x_0': self.image_size / 2 + np.random.normal(0, 3.0),
            'y_0': self.image_size / 2 + np.random.normal(0, 3.0)
        })
        
        return params

    def _get_sky_background_counts(self, filter_band, exposure_time):
        sky_mag = self.sky_magnitudes.get(filter_band, 21.2)
        collecting_area_factor = (self.primary_diameter / 8.4) ** 2
        sky_counts_per_sec = 10**((25 - sky_mag) / 2.5) * (self.pixel_scale**2) * collecting_area_factor
        
        if exposure_time < 0:
            raise ValueError("exposure_time must be non-negative")
        return sky_counts_per_sec * exposure_time

    def _add_lsst_cosmic_rays(self, image, exposure_time):
        final_image = image.copy()
        
        cr_rate_per_15s = np.random.uniform(2.0, 3.0) 
        detector_area_fraction = (self.image_size / 4096) ** 2
        expected_crs = cr_rate_per_15s * (exposure_time / 15.0) * detector_area_fraction
        expected_crs = max(1, min(expected_crs, 50))
        
        num_cosmic_rays = np.random.poisson(expected_crs)
        
        morphology_weights = ['track'] * 7 + ['spot'] * 2 + ['worm'] * 1
        
        for _ in range(num_cosmic_rays):
            cr_energy = np.clip(np.random.lognormal(np.log(20000), 0.7), 2000, 80000)
            morphology = random.choice(morphology_weights)
            
            margin = 15
            start_x = np.random.randint(margin, self.image_size - margin)
            start_y = np.random.randint(margin, self.image_size - margin)
            
            if morphology == 'spot':
                self._add_cosmic_ray_spot(final_image, start_x, start_y, cr_energy)
            elif morphology == 'worm':
                self._add_cosmic_ray_worm(final_image, start_x, start_y, cr_energy)
            else:
                self._add_cosmic_ray_track(final_image, start_x, start_y, cr_energy)
        
        return final_image, num_cosmic_rays

    def _add_cosmic_ray_spot(self, image, x, y, energy):
        spot_size = np.random.uniform(0.8, 1.2)
        y_grid, x_grid = np.mgrid[-2:3, -2:3]
        spot_kernel = np.exp(-((x_grid**2 + y_grid**2) / (2 * spot_size**2)))
        
        y_slice = slice(max(0, y-2), min(self.image_size, y+3))
        x_slice = slice(max(0, x-2), min(self.image_size, x+3))
        sy, sx = image[y_slice, x_slice].shape
        image[y_slice, x_slice] += spot_kernel[:sy, :sx] * energy

    def _add_cosmic_ray_track(self, image, start_x, start_y, energy):
        track_length = int(np.clip(np.random.exponential(12.0), 4, 40))
        angle = np.random.uniform(0, 2 * np.pi)
        
        for step in range(track_length):
            x_pos = int(start_x + step * np.cos(angle))
            y_pos = int(start_y + step * np.sin(angle))
            
            if 0 <= x_pos < self.image_size and 0 <= y_pos < self.image_size:
                energy_fraction = np.exp(-step / (track_length * 0.7)) if track_length > 0 else 1
                image[y_pos, x_pos] += energy * energy_fraction / (track_length * 0.5 + 1)

    def _add_cosmic_ray_worm(self, image, start_x, start_y, energy):
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
            
            if 0 <= x_pos < self.image_size and 0 <= y_pos < self.image_size:
                image[y_pos, x_pos] += energy / (track_length * 0.8)

    def _add_stars(self, image, galactic_latitude=30.0, filter_band='r', add_spikes=True):
        star_image = image.copy()
        
        lat_factor = 1.0 / (np.abs(np.sin(np.deg2rad(galactic_latitude))) + 0.1)
        base_density = 10000
        star_density = base_density * min(lat_factor, 5.0)
        patch_area_deg2 = (self.image_size * self.pixel_scale / 3600)**2
        expected_stars = star_density * patch_area_deg2
        num_stars = np.random.poisson(expected_stars)
        num_stars = min(num_stars, 200) 
        
        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.15), 0.4, 1.2)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.8)
        
        for _ in range(num_stars):
            zero_point_mag = 28.1
            mag = np.random.power(2.35) * 18 + 12
            amplitude = 10**(0.4 * (zero_point_mag - mag))
            
            x_pos = np.random.randint(10, self.image_size - 10)
            y_pos = np.random.randint(10, self.image_size - 10)
            
            size = 6
            star_kernel = np.zeros((2*size+1, 2*size+1), dtype=np.float32)
            star_kernel[size, size] = amplitude
            star_kernel = convolve(star_kernel, psf_kernel, boundary='extend')
            
            if add_spikes and mag < 12.0:
                spike_kernel = self._create_spike_kernel(size=30, angle_offset=45)
                spike_intensity = amplitude * 0.05
                enhanced_kernel = star_kernel + spike_intensity * spike_kernel[:star_kernel.shape[0], :star_kernel.shape[1]]
                star_kernel = enhanced_kernel
            
            y_min, y_max = max(0, y_pos - size), min(self.image_size, y_pos + size + 1)
            x_min, x_max = max(0, x_pos - size), min(self.image_size, x_pos + size + 1)
            sy, sx = star_image[y_min:y_max, x_min:x_max].shape
            star_image[y_min:y_max, x_min:x_max] += star_kernel[:sy, :sx]
        
        return star_image

    def _create_spike_kernel(self, size=30, num_spikes=4, angle_offset=45, spike_width=1.0):
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        y, x = np.mgrid[-center:center+1, -center:center+1]
        
        for i in range(num_spikes):
            angle = np.deg2rad(i * (180.0 / (num_spikes/2)) + angle_offset)
            dist_from_line = np.abs(x * np.cos(angle) + y * np.sin(angle))
            spike = np.exp(-(dist_from_line**2) / (2 * spike_width**2))
            kernel += spike
        
        return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

    def _add_blooming(self, image, saturation_limit=65000, bleed_fraction=0.1, decay_factor=0.5):
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
                    for step in range(1, self.image_size):
                        r = r_start + direction * step
                        if not (0 <= r < self.image_size) or charge_to_bleed < 1:
                            break
                        
                        bloomed_image[r, c] += charge_to_bleed
                        charge_to_bleed *= decay_factor
        
        return bloomed_image

    def _add_tle_streak(self, image, tle_line1, tle_line2, brightness=60000, width=2.5, optimal_time=None):
        # Due to missing skyfield and skimage, add a simple diagonal streak as approximation
        streak_layer = np.zeros_like(image, dtype=np.float32)
        for i in range(self.image_size):
            if i < self.image_size:
                streak_layer[i, i] += brightness
        blurred_streak = gaussian_filter(streak_layer, sigma=width/2.355)
        print("   ✅ Approximate streak added (due to missing libraries)")
        return image + blurred_streak, True

    def generate_image(self, filter_band='r', exposure_time=30, include_cosmic_rays=True, 
                      include_blooming=True, include_satellites=True, satellite_probability=0.8, 
                      tle_data=None, intersection_time=None, verbose=True):
        if verbose:
            print(f"\nLSST {filter_band}-band observation simulation started (exposure: {exposure_time}s)")

        galaxy_params = self._generate_realistic_galaxy_params(filter_band)
        allowed_keys = ["amplitude", "r_eff", "n", "x_0", "y_0", "ellip", "theta"]
        sersic_params = {k: v for k, v in galaxy_params.items() if k in allowed_keys}
        ideal_image = Sersic2D(**sersic_params)(self.x, self.y)

        ideal_image = self._add_stars(ideal_image, galactic_latitude=5.0, filter_band=filter_band, add_spikes=True)

        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.2), 0.4, 1.5)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.5)
        convolved_image = convolve(ideal_image, psf_kernel, boundary='extend')

        sky_counts = self._get_sky_background_counts(filter_band, exposure_time)
        dark_counts = 0.002 * exposure_time
        
        base_signal = convolved_image + sky_counts + dark_counts
        image_with_shot_noise = np.random.poisson(np.maximum(base_signal, 0))
        
        read_noise = np.random.normal(0, 8.0, self.image_size**2).reshape(self.image_size, self.image_size)
        noisy_image = image_with_shot_noise + read_noise
        
        final_image = noisy_image.copy()
        if include_cosmic_rays:
            final_image, cr_count = self._add_lsst_cosmic_rays(noisy_image, exposure_time)
        else:
            cr_count = 0
        
        satellite_added = False
        if include_satellites and np.random.random() < satellite_probability:
            if tle_data:
                for tle_pair in tle_data:
                    final_image, added = self._add_tle_streak(
                        final_image, tle_pair[0], tle_pair[1], optimal_time=intersection_time)
                    satellite_added = satellite_added or added
        
        if include_blooming:
            final_image = self._add_blooming(final_image)

        galaxy_type = galaxy_params.get('type', 'unknown')
        
        if verbose:
            status_msg = f"Simulation complete! ({galaxy_type} galaxy, {cr_count} cosmic rays"
            if satellite_added:
                status_msg += ", 위성 스트릭 포함"
            status_msg += ")"
            print(status_msg)

        return final_image

# --- 시뮬레이션에 사용할 샘플 TLE 데이터 ---
SAMPLE_TLE_DATA = {
    'STARLINK-G4 (53.2도)': (
        "1 64203U 25117A   25269.20698330  .00032790  00000-0  11137-2 0  9992",
        "2 64203  53.1605  83.5469 0000918  86.4920 273.6186 15.30216869 19373"
    ),
    'STARLINK-G3 (69.9도)': (
        "1 55775U 23028AL  25269.25204317  .00000871  00000-0  80769-4 0  9992",
        "2 55775  69.9979 278.9915 0003735 270.3867  89.6864 14.98339670141908"
    )
}

def create_intersection_scenario():
    simulator = LSSTAdvancedSimulator(image_size=512, seed=2025)
    
    tle1 = SAMPLE_TLE_DATA['STARLINK-G4 (53.2도)']
    tle2 = SAMPLE_TLE_DATA['STARLINK-G3 (69.9도)']
    
    intersection_time = simulator.streak_generator.find_intersection_time(tle1, tle2, max_attempts=2000)
    
    if intersection_time:
        print("\n🖼️  교차 시나리오 이미지 생성 시작...")
        intersection_image = simulator.generate_image(
            filter_band='r',
            exposure_time=30,
            include_satellites=True,
            satellite_probability=1.0,
            tle_data=[tle1, tle2],
            intersection_time=intersection_time
        )
        
        plt.figure(figsize=(10, 10))
        vmin = np.percentile(intersection_image, 1)
        vmax = np.percentile(intersection_image, 99.8)
        
        plt.imshow(intersection_image, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f"Starlink Intersection Scenario\n{intersection_time.iso}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("\n교차 시나리오 생성 실패: 주어진 조건 내에서 교차 시간을 찾지 못했습니다.")

def create_single_streak_images():
    simulator = LSSTAdvancedSimulator(image_size=512, seed=2024)
    images = {}
    for name, tle_data in SAMPLE_TLE_DATA.items():
        print(f"\n=== 단일 스트릭 생성: {name} ===")
        img = simulator.generate_image(filter_band='r', exposure_time=30, tle_data=[tle_data])
        images[name] = img
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for i, (name, img) in enumerate(images.items()):
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99.5)
        axes[i].imshow(img, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{name}", fontsize=14)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("LSST Starlink Streak Simulator")
    print("=" * 50)
    
    create_intersection_scenario()
    
    print("\n" + "=" * 50)
    
    print("🛰️  비교를 위해 개별 위성 스트릭 이미지를 생성합니다.")
    create_single_streak_images()
    
    print("\n✅ 시뮬레이션 완료!")