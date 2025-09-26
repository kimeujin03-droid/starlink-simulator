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
    위성 스트릭(streak) 생성 및 관리를 위한 전문 클래스입니다.
    LSSTAdvancedSimulator로부터 WCS(좌표계), 관측자 정보 등을 받아
    현실적인 위성 궤적을 계산하고 그리는 역할을 담당합니다.
    """
    def __init__(self, simulator):
        # --- 초기화 함수 ---
        # simulator: LSSTAdvancedSimulator의 인스턴스. 이미지 크기, WCS 등 시뮬레이터의 정보에 접근하기 위해 필요합니다.
        self.sim = simulator
        # ts: skyfield 라이브러리의 시간 척도(timescale) 객체. 천문 계산에 필요한 표준 시간 시스템입니다.
        self.ts = load.timescale()

    def find_optimal_observation_time(self, tle_line1, tle_line2, search_days=365, max_attempts=1000, return_all=False):
        """
        주어진 TLE 데이터의 위성이 특정 기간 내에 관측 시야를 통과하는 '최적의 시간'을 찾습니다.

        Args:
            tle_line1 (str): TLE 데이터의 첫 번째 줄.
            tle_line2 (str): TLE 데이터의 두 번째 줄.
            search_days (int): 탐색할 기간 (일 단위). 기본값은 1년(365일)입니다.
            max_attempts (int): 최적 시간을 찾기 위해 무작위로 시도할 횟수.
            return_all (bool): True일 경우, 찾은 모든 유효 시간을 리스트로 반환합니다. False일 경우, 그중 하나를 무작위로 선택해 반환합니다.

        Returns:
            astropy.time.Time 또는 list 또는 None: 찾은 관측 시간. 못 찾으면 None을 반환합니다.
        """
        # TLE 데이터를 이용해 skyfield의 위성 객체를 생성합니다.
        satellite = EarthSatellite(tle_line1, tle_line2, 'SAT', self.ts)
        # 탐색 시작 기준 시간을 설정합니다. (과거 1년 중 임의의 시점부터 시작하여 매번 다른 결과를 얻도록 함)
        now = Time.now() - TimeDelta(np.random.uniform(0, 365*24*3600), format='sec') # 과거 1년부터 탐색 시작
        # 찾은 유효 시간들을 저장할 리스트입니다.
        found_times = []

        # 지정된 횟수만큼 탐색을 반복합니다.
        for _ in range(max_attempts):
            # 탐색 기간 내에서 무작위로 시간을 선택합니다.
            random_hours = np.random.uniform(0, search_days * 24)
            test_time = now + TimeDelta(random_hours * 3600, format='sec')
            exposure_duration = TimeDelta(30, format='sec')

            # 선택된 시간(test_time)부터 30초 노출 동안 100개의 지점에서 위성의 위치를 계산합니다.
            times = self.ts.linspace(self.ts.from_astropy(test_time),
                                     self.ts.from_astropy(test_time + exposure_duration),
                                     100)

            # 1. 지구 중심에서의 위성 위치 계산
            geocentric = satellite.at(times)
            # 2. LSST 천문대(관측자)의 위치 계산
            observer_at_time = self.sim.observer.at(times)
            # 3. 관측자 중심에서의 위성 위치 계산 (실제 하늘에서 보이는 위치)
            topocentric = geocentric - observer_at_time
            # 4. 하늘 좌표(적경, 적위)로 변환
            ra, dec, _ = topocentric.radec()

            try:
                # 하늘 좌표를 시뮬레이션 이미지의 픽셀 좌표(px, py)로 변환합니다.
                px, py = self.sim.wcs.world_to_pixel_values(ra.degrees, dec.degrees) # type: ignore
                # 픽셀 좌표가 이미지 영역(+약간의 여백) 안에 들어오는지 확인합니다.
                valid = (px >= -50) & (px < self.sim.image_size + 50) & \
                        (py >= -50) & (py < self.sim.image_size + 50)

                # 궤적 중 20개 이상의 점이 시야에 들어오면 유효한 통과로 간주합니다.
                if np.sum(valid) >= 20:
                    # 모든 시간을 찾을 필요가 없다면, 첫 번째로 찾은 시간을 바로 반환합니다.
                    if not return_all:
                        print(f"   ✅ 첫 번째 유효 관측 시간 발견: {test_time.iso}")
                        return test_time
                    # 모든 시간을 찾아야 한다면, 찾은 시간을 리스트에 추가합니다.
                    found_times.append(test_time)
            except Exception:
                # 좌표 변환 등에서 오류가 발생하면 무시하고 다음 시도를 계속합니다.
                continue

        # 모든 시도 후, 찾은 시간들의 처리
        if return_all:
            # return_all=True이면, 찾은 시간 전체 리스트를 반환합니다.
            print(f"   🔍 {len(found_times)}개의 유효 시간을 찾았습니다.")
            return found_times
        elif found_times:
            # 찾은 시간들 중에서 무작위로 하나를 선택하여 반환합니다.
            selected_time = random.choice(found_times)
            print(f"   ✅ {len(found_times)}개의 유효 시간 중 무작위 선택: {selected_time.iso}")
            return selected_time
        else:
            # 결국 유효한 시간을 찾지 못하면 None을 반환합니다.
            print(f"   ❌ {max_attempts}번 시도 후에도 적절한 시간을 찾지 못했습니다.")
            return None

    def find_intersection_time(self, tle_pair1, tle_pair2, search_days=365, max_attempts=1000):
        """
        두 개의 위성이 '동시에' 관측 시야를 통과하는 매우 희귀한 '교차 시간'을 찾습니다.
        이것이 이 시뮬레이터의 핵심 기능 중 하나입니다.

        Args:
            tle_pair1 (tuple): 첫 번째 위성의 TLE 데이터 (line1, line2).
            tle_pair2 (tuple): 두 번째 위성의 TLE 데이터 (line1, line2).
            search_days (int): 탐색 기간.
            max_attempts (int): 탐색 시도 횟수.

        Returns:
            astropy.time.Time 또는 None: 찾은 교차 시간. 못 찾으면 None.
        """
        print("\n🛰️  두 위성의 교차 시간 탐색 시작...")
        
        # [1단계] 첫 번째 위성이 관측 시야를 통과하는 '모든' 가능한 시간대를 찾아 리스트로 만듭니다.
        print(f"   1. 위성 1의 통과 시간 탐색 (최대 {max_attempts}회 시도)...")
        sat1_times = self.find_optimal_observation_time(
            tle_pair1[0], tle_pair1[1], search_days=search_days, max_attempts=max_attempts, return_all=True
        )
        # 만약 첫 번째 위성조차 통과하는 시간을 찾지 못하면, 교차는 불가능하므로 탐색을 중단합니다.
        if not sat1_times:
            print("   ❌ 위성 1의 통과 시간을 찾지 못해 교차 탐색을 중단합니다.")
            return None

        # [2단계] 1단계에서 찾은 시간 후보들 각각에 대해, '두 번째 위성'도 그 시간에 통과하는지 확인합니다.
        print(f"   2. 위성 2가 {len(sat1_times)}개의 시간대에 동시 통과하는지 확인 중...")
        satellite2 = EarthSatellite(tle_pair2[0], tle_pair2[1], 'SAT2', self.ts)
        
        # 효율적인 탐색을 위해 시간 순서대로 정렬하여 확인합니다.
        for t in sorted(sat1_times):
            exposure_duration = TimeDelta(30, format='sec')
            times = self.ts.linspace(self.ts.from_astropy(t), self.ts.from_astropy(t + exposure_duration), 100)
            
            # 두 번째 위성의 궤적을 계산합니다.
            geocentric = satellite2.at(times)
            observer_at_time = self.sim.observer.at(times)
            topocentric = geocentric - observer_at_time
            ra, dec, _ = topocentric.radec()

            # 픽셀 좌표로 변환하여 시야에 들어오는지 확인합니다.
            px, py = self.sim.wcs.world_to_pixel_values(ra.degrees, dec.degrees) # type: ignore
            valid = (px >= -50) & (px < self.sim.image_size + 50) & (py >= -50) & (py < self.sim.image_size + 50)

            # 만약 두 번째 위성도 시야를 통과한다면, 이것이 바로 '교차 시간'입니다!
            if np.sum(valid) >= 20:
                print(f"   🔥🔥🔥 교차 시간 발견! {t.iso}")
                return t # 첫 번째로 찾은 교차 시간을 즉시 반환합니다.
        
        # 모든 후보 시간을 확인했지만 교차점을 찾지 못한 경우입니다.
        print("   ❌ 모든 후보 시간에서 교차점을 찾지 못했습니다.")
        return None

class LSSTAdvancedSimulator:
    """
    LSST/Vera C. Rubin Observatory advanced image simulator with satellite streaks.
    """
    
    def __init__(self, image_size=512, seed=42):
        """
        Initialize the simulator.
        
        Args:
            image_size (int): Size of generated image in pixels
            seed (int): Random seed for reproducibility
        """
        self.image_size = image_size
        self.pixel_scale = 0.2  # LSST pixel scale: 0.2"/pixel
        self.primary_diameter = 6.423  # Primary mirror effective diameter (m)
        self.field_of_view = 9.6  # deg²
        
        # LSST system parameters
        self.standard_exposure = 15.0  # Standard visit exposure time (sec)
        self.visits_per_observation = 2  # Exposures per visit
        self.read_noise_range = (5.4, 6.2)  # Read noise range (e-)
        self.gain_range = (1.5, 1.7)  # System gain (e-/ADU)
        self.r_band_5sigma_depth = 24.7  # r-band 5σ single visit depth
        
        self.y, self.x = np.mgrid[:image_size, :image_size].astype(np.float32)
        np.random.seed(seed)
        random.seed(seed)
        
        # LSST sky brightness (mag/arcsec²)
        self.sky_magnitudes = {
            'u': 22.9, 'g': 22.3, 'r': 21.2, 
            'i': 20.5, 'z': 19.6, 'y': 18.6
        }
        
        # Zero points by filter (30s exposure basis)
        self.zero_points_30s = {
            'u': 27.0, 'g': 28.3, 'r': 28.1, 
            'i': 27.9, 'z': 27.4, 'y': 26.5
        }
        
        # Initialize WCS info
        self.wcs = self._create_default_wcs()
        
        # Observer location (LSST location - Cerro Pachon, Chile)
        # LSST 천문대의 실제 지리적 위치(위도, 경도, 고도)를 Topos 객체로 설정합니다.
        self.observer = Topos('30.2444 S', '70.7494 W', elevation_m=2663)

        # 위성 스트릭 생성을 전담하는 SatelliteStreak 클래스의 인스턴스를 생성합니다.
        self.streak_generator = SatelliteStreak(self)
        
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        print(f"LSST Advanced Simulator initialized (image size: {image_size}x{image_size})")

    def _create_default_wcs(self):
        """Create default WCS information for a random sky area"""
        # WCS(World Coordinate System)는 이미지의 픽셀 좌표(x, y)와
        # 실제 하늘의 천구 좌표(적경, 적위)를 변환해주는 지도와 같은 역할을 합니다.
        wcs = WCS(naxis=2)
        
        # Set random sky area (near equator for better satellite coverage)
        # 위성 통과 확률을 높이기 위해 적도 근처의 임의의 하늘 영역을 관측 대상으로 설정합니다.
        ra_center = np.random.uniform(0, 360)  # degrees
        dec_center = np.random.uniform(-30, 30)  # degrees
        
        # WCS 파라미터 설정
        wcs.wcs.crpix = [self.image_size/2, self.image_size/2]  # 기준점 픽셀 (이미지 중앙)
        wcs.wcs.crval = [ra_center, dec_center]  # 기준점 픽셀의 하늘 좌표 (RA, Dec)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        # 픽셀 스케일 설정 (LSST의 0.2 arcsec/pixel을 도 단위로 변환)
        wcs.wcs.cdelt = [-self.pixel_scale/3600, self.pixel_scale/3600]  # Pixel scale (deg/pixel)
        
        return wcs

    def set_field_center(self, ra, dec):
        """Set field center coordinates"""
        self.wcs.wcs.crval = [ra, dec]
        print(f"Field center set: RA={ra:.2f}°, Dec={dec:.2f}°")

    def _generate_realistic_galaxy_params(self, filter_band='r'):
        """Generate realistic galaxy parameters based on observations"""
        # LSST 관측 데이터를 기반으로 현실적인 은하의 파라미터를 생성합니다.
        
        # 1. 은하의 등급(밝기)을 무작위로 결정하고, 이를 시뮬레이션에서의 밝기(amplitude)로 변환합니다.
        zero_point_mag = self.zero_points_30s.get(filter_band, 28.1) 
        mag_r = np.random.uniform(18.0, 24.5)
        amplitude = 10**(0.4 * (zero_point_mag - mag_r))
        
        galaxy_type_prob = np.random.random()
        # 2. 은하의 형태(타원, 나선, 불규칙)를 확률에 따라 결정하고, 각 형태에 맞는 파라미터를 설정합니다.
        if galaxy_type_prob < 0.3:  # Elliptical (30%)
            params = {
                'type': 'elliptical',
                'n': np.clip(np.random.normal(4.0, 0.5), 0.5, 8.0),
                'r_eff_kpc': np.random.lognormal(np.log(4.0), 0.3),
                'ellip': np.clip(np.random.beta(2, 2) * 0.8, 0.0, 0.9),
                'amplitude': amplitude * 1.2
            }
        elif galaxy_type_prob < 0.8:  # Spiral (50%)
            params = {
                'type': 'spiral',
                'n': np.clip(np.random.normal(1.0, 0.2), 0.5, 8.0),
                'r_eff_kpc': np.random.lognormal(np.log(4.5), 0.4),
                'ellip': np.clip(np.random.beta(1.5, 3) * 0.6, 0.0, 0.9),
                'amplitude': amplitude
            }
        else:  # Irregular (20%)
            params = {
                'type': 'irregular',
                'n': np.random.uniform(0.5, 2.0),
                'r_eff_kpc': np.random.lognormal(np.log(2.5), 0.6),
                'ellip': np.clip(np.random.beta(1, 1) * 0.7, 0.0, 0.9),
                'amplitude': amplitude * 0.8
            }
        
        # 3. 은하의 거리(적색편이) 효과를 대략적으로 시뮬레이션하여 크기를 조절합니다.
        redshift_factor = np.random.uniform(0.8, 1.5)
        r_eff_pixels = params['r_eff_kpc'] * 5.0 / redshift_factor
        
        # 4. 최종 파라미터들을 업데이트하고, 값들이 비정상적으로 크거나 작아지지 않도록 범위를 제한(clip)합니다.
        params.update({
            'amplitude': np.clip(params['amplitude'], 100, 5e5),
            'r_eff': np.clip(r_eff_pixels, 1.0, self.image_size // 6),
            'theta': np.random.uniform(0, np.pi),
            # 은하의 중심 위치를 이미지 중앙에서 약간 벗어나도록 설정합니다.
            'x_0': self.image_size / 2 + np.random.normal(0, 3.0),
            'y_0': self.image_size / 2 + np.random.normal(0, 3.0)
        })
        
        return params

    def _get_sky_background_counts(self, filter_band, exposure_time):
        """Calculate LSST sky background brightness in electron counts"""
        # 밤하늘 자체가 완전히 검지 않고 희미하게 빛나는 '하늘 배경 밝기'를 계산합니다.
        sky_mag = self.sky_magnitudes.get(filter_band, 21.2)
        collecting_area_factor = (self.primary_diameter / 8.4) ** 2
        sky_counts_per_sec = 10**((25 - sky_mag) / 2.5) * (self.pixel_scale**2) * collecting_area_factor
        
        if exposure_time < 0:
            raise ValueError("exposure_time must be non-negative")
        return sky_counts_per_sec * exposure_time

    def _add_lsst_cosmic_rays(self, image, exposure_time):
        """Add cosmic ray effects based on LSST actual rates"""
        # 우주에서 날아온 고에너지 입자(우주선, Cosmic Ray)가 CCD 센서에 부딪혀 생기는 노이즈를 추가합니다.
        final_image = image.copy()
        
        # LSST의 실제 우주선 검출률을 바탕으로, 주어진 노출 시간 동안 나타날 우주선의 예상 개수를 계산합니다.
        cr_rate_per_15s = np.random.uniform(2.0, 3.0) 
        detector_area_fraction = (self.image_size / 4096) ** 2
        expected_crs = cr_rate_per_15s * (exposure_time / 15.0) * detector_area_fraction
        expected_crs = max(1, min(expected_crs, 50))
        
        # 푸아송 분포를 이용해 실제 나타날 우주선의 개수를 무작위로 결정합니다.
        num_cosmic_rays = np.random.poisson(expected_crs)
        
        morphology_weights = ['track'] * 7 + ['spot'] * 2 + ['worm'] * 1
        
        # 각 우주선에 대해 형태(직선, 점, 벌레 모양)와 에너지를 무작위로 결정하고 이미지에 그립니다.
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
            else:  # track
                self._add_cosmic_ray_track(final_image, start_x, start_y, cr_energy)
        
        return final_image, num_cosmic_rays

    def _add_cosmic_ray_spot(self, image, x, y, energy):
        """Add point-like cosmic ray"""
        # 점 모양의 우주선을 추가합니다. 가우시안 커널을 이용해 점을 표현합니다.
        spot_size = np.random.uniform(0.8, 1.2)
        y_grid, x_grid = np.mgrid[-2:3, -2:3]
        spot_kernel = np.exp(-((x_grid**2 + y_grid**2) / (2 * spot_size**2)))
        
        y_slice = slice(max(0, y-2), min(self.image_size, y+3))
        x_slice = slice(max(0, x-2), min(self.image_size, x+3))
        sy, sx = image[y_slice, x_slice].shape
        image[y_slice, x_slice] += spot_kernel[:sy, :sx] * energy

    def _add_cosmic_ray_track(self, image, start_x, start_y, energy):
        """Add linear track cosmic ray"""
        # 직선 궤적 모양의 우주선을 추가합니다.
        track_length = int(np.clip(np.random.exponential(12.0), 4, 40))
        angle = np.random.uniform(0, 2 * np.pi)
        
        for step in range(track_length):
            x_pos = int(start_x + step * np.cos(angle))
            y_pos = int(start_y + step * np.sin(angle))
            
            if 0 <= x_pos < self.image_size and 0 <= y_pos < self.image_size:
                energy_fraction = np.exp(-step / (track_length * 0.7)) if track_length > 0 else 1
                image[y_pos, x_pos] += energy * energy_fraction / (track_length * 0.5 + 1)

    def _add_cosmic_ray_worm(self, image, start_x, start_y, energy):
        """Add worm-like curved cosmic ray"""
        # 벌레처럼 구불구불한 모양의 우주선을 추가합니다. 사인 함수를 이용해 곡선 경로를 만듭니다.
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
        """Add stars based on galactic coordinate stellar density"""
        # 이미지에 배경 별들을 추가합니다.
        star_image = image.copy()
        
        # 은하수(galactic_latitude=0)에 가까울수록 별이 많아지는 것을 시뮬레이션합니다.
        lat_factor = 1.0 / (np.abs(np.sin(np.deg2rad(galactic_latitude))) + 0.1)
        base_density = 10000
        star_density = base_density * min(lat_factor, 5.0)
        patch_area_deg2 = (self.image_size * self.pixel_scale / 3600)**2
        expected_stars = star_density * patch_area_deg2
        num_stars = np.random.poisson(expected_stars)
        num_stars = min(num_stars, 200) # 별이 너무 많아지지 않도록 최대 200개로 제한합니다.
        
        # 대기의 흔들림(seeing) 효과를 시뮬레이션하기 위한 PSF(Point Spread Function) 커널을 생성합니다.
        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.15), 0.4, 1.2)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.8)
        
        for _ in range(num_stars):
            # 별의 밝기(등급)를 무작위로 결정합니다. (어두운 별이 더 많도록 power-law 분포 사용)
            zero_point_mag = 28.1
            mag = np.random.power(2.35) * 18 + 12
            amplitude = 10**(0.4 * (zero_point_mag - mag))
            
            # 별의 위치를 무작위로 결정합니다.
            x_pos = np.random.randint(10, self.image_size - 10)
            y_pos = np.random.randint(10, self.image_size - 10)
            
            # 이상적인 점광원(별)을 생성하고, PSF 커널을 적용하여 흐릿하게 만듭니다.
            size = 6
            star_kernel = np.zeros((2*size+1, 2*size+1), dtype=np.float32)
            star_kernel[size, size] = amplitude
            star_kernel = convolve(star_kernel, psf_kernel, boundary='extend')
            
            # 매우 밝은 별(12등급 미만)일 경우, 망원경 구조물에 의해 생기는 회절 스파이크(십자 모양 빛)를 추가합니다.
            if add_spikes and mag < 12.0:
                spike_kernel = self._create_spike_kernel(size=30, angle_offset=45)
                spike_intensity = amplitude * 0.05
                enhanced_kernel = star_kernel + spike_intensity * spike_kernel[:star_kernel.shape[0], :star_kernel.shape[1]]
                star_kernel = enhanced_kernel
            
            # 완성된 별 이미지를 전체 이미지에 더합니다.
            y_min, y_max = max(0, y_pos - size), min(self.image_size, y_pos + size + 1)
            x_min, x_max = max(0, x_pos - size), min(self.image_size, x_pos + size + 1)
            sy, sx = star_image[y_min:y_max, x_min:x_max].shape
            star_image[y_min:y_max, x_min:x_max] += star_kernel[:sy, :sx]
        
        return star_image

    def _create_spike_kernel(self, size=30, num_spikes=4, angle_offset=45, spike_width=1.0):
        """Create LSST telescope structure-based diffraction spike kernel"""
        # 밝은 별 주위에 나타나는 십자 모양의 회절 스파이크를 만드는 커널을 생성합니다.
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
        """Apply realistic exponential decay model for LSST CCD blooming effects"""
        # CCD 센서의 '블루밍' 효과를 시뮬레이션합니다.
        # 매우 밝은 빛이 들어오면 픽셀이 담을 수 있는 전하량(saturation_limit)을 초과하여,
        # 넘친 전하가 주변 픽셀(주로 세로 방향)로 흘러들어가는 현상입니다.
        bloomed_image = image.copy()
        # 포화된 픽셀들의 좌표를 찾습니다.
        saturated_coords = np.argwhere(bloomed_image > saturation_limit)
        
        if saturated_coords.shape[0] == 0:
            return bloomed_image
        
        # 각 세로줄(column)별로 블루밍을 처리합니다.
        for c in np.unique(saturated_coords[:, 1]):
            col_saturated_rows = sorted(saturated_coords[saturated_coords[:, 1] == c][:, 0])
            
            for r_start in col_saturated_rows:
                # 포화 한계를 넘은 초과 전하량을 계산합니다.
                excess_charge = (bloomed_image[r_start, c] - saturation_limit) * bleed_fraction
                bloomed_image[r_start, c] = saturation_limit
                
                # 위, 아래 방향으로 초과 전하를 흘려보냅니다.
                for direction in [-1, 1]:
                    charge_to_bleed = excess_charge / 2.0
                    for step in range(1, self.image_size):
                        r = r_start + direction * step
                        if not (0 <= r < self.image_size) or charge_to_bleed < 1:
                            break
                        
                        # 거리가 멀어질수록 흘러들어가는 전하량이 감소하도록(decay_factor) 처리합니다.
                        bloomed_image[r, c] += charge_to_bleed
                        charge_to_bleed *= decay_factor
        
        return bloomed_image

    def _add_tle_streak(self, image, tle_line1, tle_line2, brightness=60000, width=2.5, optimal_time=None):
        """
        TLE 데이터를 기반으로 위성 스트릭을 이미지에 추가합니다.

        Args:
            image (ndarray): 스트릭을 추가할 원본 이미지.
            tle_line1, tle_line2 (str): 위성의 TLE 데이터.
            brightness (float): 스트릭의 기본 밝기.
            width (float): 스트릭의 두께.
            optimal_time (astropy.time.Time, optional): 스트릭을 그릴 시간을 직접 지정.
                                                        None이면 자동으로 시간을 탐색합니다. (교차 시나리오에서 사용)

        Returns:
            tuple: (스트릭이 추가된 이미지, 성공 여부 bool)
        """
        try:
            # 만약 스트릭을 그릴 시간이 지정되지 않았다면, 최적 시간을 자동으로 탐색합니다.
            if optimal_time is None:
                optimal_time = self.streak_generator.find_optimal_observation_time(tle_line1, tle_line2)
            if optimal_time is None:
                print("   TLE 기반 스트릭 생성 실패. 스트릭을 추가하지 않습니다.")
                return image, False

            # 지정된 시간 동안의 위성 궤적을 계산합니다.
            ts = self.streak_generator.ts
            satellite = EarthSatellite(tle_line1, tle_line2, 'SAT', ts)
            exposure_duration = TimeDelta(30, format='sec')
            num_steps = int(exposure_duration.sec * 20)
            times = ts.linspace(ts.from_astropy(optimal_time),
                               ts.from_astropy(optimal_time + exposure_duration),
                               num_steps)

            geocentric = satellite.at(times)
            observer_at_time = self.observer.at(times)
            topocentric = geocentric - observer_at_time
            ra, dec, distance = topocentric.radec()

            # 계산된 궤적을 이미지의 픽셀 좌표로 변환합니다.
            pixel_coords = self.wcs.world_to_pixel_values(ra.degrees, dec.degrees) # type: ignore
            px, py = pixel_coords[0], pixel_coords[1]

            # 궤적이 이미지 경계 내에 있는지 확인합니다.
            valid = (px >= 0) & (px < self.image_size) & (py >= 0) & (py < self.image_size)

            if not np.any(valid):
                print("   위성이 시야에 들어오지 않습니다. 스트릭을 추가하지 않습니다.")
                return image, False

            # 스트릭을 그릴 빈 레이어를 생성합니다.
            streak_layer = np.zeros_like(image, dtype=np.float32)
            
            # 위성과의 거리에 따라 밝기를 조절합니다. (가까울수록 밝게)
            valid_distances = distance.km[valid]
            distance_factor = np.clip(1500.0 / np.mean(valid_distances), 0.3, 5.0) if len(valid_distances) > 0 else 1.0
            adjusted_brightness = brightness * distance_factor

            valid_px = px[valid]
            valid_py = py[valid]

            # 궤적의 각 점들을 직선으로 연결하여 스트릭을 그립니다.
            for i in range(len(valid_px) - 1):
                # line_aa를 사용해 부드러운(anti-aliased) 선을 그립니다.
                rr, cc, val = line_aa(int(valid_py[i]), int(valid_px[i]),
                                     int(valid_py[i+1]), int(valid_px[i+1]))
                mask = (rr >= 0) & (rr < self.image_size) & (cc >= 0) & (cc < self.image_size)
                streak_layer[rr[mask], cc[mask]] = np.maximum(streak_layer[rr[mask], cc[mask]],
                                                             val[mask] * adjusted_brightness)

            blurred_streak = gaussian_filter(streak_layer, sigma=width/2.355)
            # 최종적으로 스트릭 레이어를 원본 이미지에 더합니다.
            print(f"   ✅ TLE 기반 스트릭 추가 완료 ({np.sum(valid)} 지점, 밝기: {adjusted_brightness:.0f})")
            return image + blurred_streak, True

        except Exception as e:
            print(f"   ⚠️ TLE 스트릭 계산 중 오류 발생: {e}")
            return image, False

    def generate_image(self, filter_band='r', exposure_time=30, include_cosmic_rays=True, 
                      include_blooming=True, include_satellites=True, satellite_probability=0.8, 
                      tle_data=None, intersection_time=None, verbose=True):
        """
        모든 구성 요소를 종합하여 최종적인 단일 LSST 관측 이미지를 생성하는 메인 함수입니다.
        이 함수는 이미지 생성의 전체 과정을 순서대로 지휘합니다.
        """
        if verbose:
            print(f"\nLSST {filter_band}-band observation simulation started (exposure: {exposure_time}s)")

        # [1단계] 이상적인 은하 생성
        # Sersic 프로파일을 이용해 수학적으로 완벽한 은하 이미지를 만듭니다.
        galaxy_params = self._generate_realistic_galaxy_params(filter_band)
        allowed_keys = ["amplitude", "r_eff", "n", "x_0", "y_0", "ellip", "theta"]
        sersic_params = {k: v for k, v in galaxy_params.items() if k in allowed_keys}
        ideal_image = Sersic2D(**sersic_params)(self.x, self.y)

        # [2단계] 배경 별 추가
        # 은하수 위치를 고려하여 현실적인 개수의 별들을 이미지에 추가합니다.
        ideal_image = self._add_stars(ideal_image, galactic_latitude=5.0, filter_band=filter_band, add_spikes=True)

        # [3단계] 대기 및 망원경 효과 적용 (PSF)
        # 지구 대기의 흔들림(seeing)과 망원경의 광학적 한계로 인해 이미지가 흐려지는 효과를 적용합니다.
        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.2), 0.4, 1.5)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.5)
        convolved_image = convolve(ideal_image, psf_kernel, boundary='extend')

        # [4단계] CCD 노이즈 추가
        # 실제 CCD 센서에서 발생하는 여러 종류의 노이즈를 추가합니다.
        # 4-1. 하늘 배경 노이즈
        sky_counts = self._get_sky_background_counts(filter_band, exposure_time)
        # 4-2. 암전류 노이즈 (센서 자체의 열로 인해 발생)
        dark_counts = 0.002 * exposure_time
        
        base_signal = convolved_image + sky_counts + dark_counts
        # 4-3. 샷 노이즈 (빛 입자의 무작위성으로 인해 발생)
        image_with_shot_noise = np.random.poisson(np.maximum(base_signal, 0))
        
        # 4-4. 리드 노이즈 (센서 데이터를 읽어올 때 발생)
        read_noise = np.random.normal(0, 8.0, self.image_size**2).reshape(self.image_size, self.image_size)
        noisy_image = image_with_shot_noise + read_noise
        
        # [5단계] 우주선(Cosmic Rays) 추가
        final_image = noisy_image.copy()
        if include_cosmic_rays:
            final_image, cr_count = self._add_lsst_cosmic_rays(noisy_image, exposure_time)
        else:
            cr_count = 0
        
        # [6단계] 위성 스트릭 추가
        satellite_added = False
        if include_satellites and np.random.random() < satellite_probability:
            if tle_data:
                # tle_data가 리스트 형태이므로, 리스트에 있는 모든 위성에 대해 스트릭을 그립니다.
                # 교차 시나리오의 경우, tle_data에 두 개의 위성 정보가 들어있고, intersection_time이 지정됩니다.
                for tle_pair in tle_data:
                    final_image, added = self._add_tle_streak(
                        final_image, tle_pair[0], tle_pair[1], optimal_time=intersection_time)
                    satellite_added = satellite_added or added
        
        # [7단계] 블루밍(Blooming) 효과 추가
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
    'STARLINK-G4 (53.2도)': ( # 53.2도 경사각을 가져 LSST 관측 가능성이 높은 위성
        "1 53099U 22082CH  24155.50000000  .00002100  00000+0  42000-3 0  9991",
        "2 53099  53.2173 211.2053 0001500 105.3000 254.8000 15.08200000 98001"
    ),
    'STARLINK-G3 (69.9도)': ( # 69.9도 경사각을 가져 남반구 관측에 더 유리한 위성
        "1 56814U 23091K   24155.50000000  .00002500  00000+0  48000-3 0  9992",
        "2 56814  69.9980 180.0000 0001200 135.0000 225.1000 15.11500000 45008"
    )
}

def main():
    """
    LSSTAdvancedSimulator를 사용하여 Starlink 위성 스트릭이 포함된
    현실적인 천문 이미지를 생성하고 화면에 보여줍니다.
    """
    print("🚀 고급 LSST 시뮬레이터를 사용하여 이미지 생성 시작...")

    # 1. 시뮬레이터 인스턴스 생성
    # seed 값을 바꾸면 매번 다른 은하, 별, 노이즈 패턴이 생성됩니다.
    simulator = LSSTAdvancedSimulator(image_size=512, seed=2024)

    # 2. 시뮬레이션할 Starlink 위성 TLE 데이터 가져오기
    # 함수모음.py에 정의된 샘플 데이터 중 하나를 선택합니다.
    tle_data = SAMPLE_TLE_DATA['STARLINK-G4 (53.2도)']
    
    # 3. 위성 스트릭을 포함한 최종 이미지 생성
    # generate_image 함수 하나만 호출하면 모든 복잡한 과정이 내부적으로 처리됩니다.
    sim_image = simulator.generate_image(
        filter_band='r',
        exposure_time=30,
        include_satellites=True,
        satellite_probability=1.0, # 100% 확률로 스트릭 생성
        tle_data=[tle_data]        # tle_data는 리스트 형태로 전달해야 합니다.
    )

    # 4. 생성된 이미지 시각화
    if sim_image is not None:
        plt.figure(figsize=(10, 10))
        # vmin, vmax를 조절하여 노이즈 속 천체와 스트릭이 잘 보이도록 대비를 설정합니다.
        vmin = np.percentile(sim_image, 1)
        vmax = np.percentile(sim_image, 99.8)
        
        plt.imshow(sim_image, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        plt.title('LSST Simulated Image with Starlink Streak', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# --- 메인 실행 블록 ---
# 이 스크립트 파일을 직접 실행했을 때만 아래 코드가 동작합니다.
if __name__ == '__main__':
    print("LSST Starlink Streak Simulator")
    print("=" * 50)
    main()
    print("\n✅ 시뮬레이션 완료!")