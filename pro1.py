import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, Moffat2DKernel

class LSSTAdvancedSimulator:
    """
    LSST/Vera C. Rubin Observatory의 관측 환경을 시뮬레이션하는 고급 이미지 시뮬레이터.
    
    주요 기능:
    - 현실적인 은하 파라미터 생성 (타원, 나선, 불규칙 은하)
    - 다양한 형태의 우주선 시뮬레이션 (점, 궤적, 구불구불한 형태)
    - LSST 관측 조건 반영 (PSF, 노이즈, 배경)
    - 별 및 밝은 천체 효과 (스파이크, 블루밍)
    """
    
    def __init__(self, image_size=512, seed=42):
        """
        시뮬레이터 초기화.
        
        Args:
            image_size (int): 생성할 이미지의 크기 (픽셀 단위)
            seed (int): 재현성을 위한 랜덤 시드
        """
        self.image_size = image_size
        self.pixel_scale = 0.2  # LSST 픽셀 스케일: 0.2"/pixel
        self.y, self.x = np.mgrid[:image_size, :image_size].astype(np.float64)
        
        # 랜덤 시드 고정
        np.random.seed(seed)
        random.seed(seed)
        
        # LSST 필터별 하늘 배경 밝기 (mag/arcsec²)
        self.sky_magnitudes = {
            'u': 22.9, 'g': 22.3, 'r': 21.2, 
            'i': 20.5, 'z': 19.6, 'y': 18.6
        }
        
        print(f"🔭 LSST 고급 시뮬레이터 초기화 (이미지 크기: {image_size}x{image_size})")

    def _generate_realistic_galaxy_params(self, filter_band='r'):
    
        mag_r = np.random.uniform(20.0, 26.0)
        amplitude = 10**((25 - mag_r) / 2.5) * 100

        galaxy_type_prob = np.random.random()

        if galaxy_type_prob < 0.3:  # 타원 은하
            params = {
                'type': 'elliptical',
                'n': np.random.normal(4.0, 0.8),
                'r_eff': np.random.lognormal(np.log(15), 0.4),
                'ellip': np.random.beta(2, 2) * 0.8,
                'amplitude': amplitude * 1.5
            }
        elif galaxy_type_prob < 0.8:  # 나선 은하
            params = {
                'type': 'spiral',
                'n': np.random.normal(1.0, 0.3),
                'r_eff': np.random.lognormal(np.log(25), 0.5),
                'ellip': np.random.beta(1.5, 3) * 0.6,
                'amplitude': amplitude
            }
        else:  # 불규칙 은하
            params = {
                'type': 'irregular',
                'n': np.random.uniform(0.5, 2.0),
                'r_eff': np.random.lognormal(np.log(10), 0.6),
                'ellip': np.random.beta(1, 1) * 0.7,
                'amplitude': amplitude * 0.7
            }

        # 안전한 값으로 클리핑
        params['amplitude'] = np.clip(params['amplitude'], 100, 5e5)

        params.update({
            'n': np.clip(params['n'], 0.3, 8.0),
            'r_eff': np.clip(params['r_eff'], 2.0, self.image_size // 4),
            'ellip': np.clip(params['ellip'], 0.0, 0.95),
            'theta': np.random.uniform(0, np.pi),
            'x_0': self.image_size / 2 + np.random.normal(0, 2.0) / self.pixel_scale,
            'y_0': self.image_size / 2 + np.random.normal(0, 2.0) / self.pixel_scale
        })

        return params

    def _get_sky_background_counts(self, filter_band, exposure_time):
        """
        필터별 하늘 배경 밝기를 전자 수로 계산합니다.
        
        Args:
            filter_band (str): 관측 필터 ('u', 'g', 'r', 'i', 'z', 'y')
            exposure_time (float): 노출 시간 (초)
            
        Returns:
            float: 픽셀당 배경 전자 수
        """
        sky_mag = self.sky_magnitudes.get(filter_band, 21.2)
        sky_counts_per_sec = 10**((25 - sky_mag) / 2.5) * (self.pixel_scale**2)
        return sky_counts_per_sec * exposure_time

    def _add_lsst_cosmic_rays(self, image, exposure_time):

        final_image = image.copy()

    # 검출기 면적 (대략, cm²)
        detector_area_cm2 = (self.image_size * 0.01)**2  # 픽셀당 0.01cm 가정
        cr_rate = 2.5  # CR/cm²/min @ Cerro Pachón
        expected_crs = cr_rate * detector_area_cm2 * (exposure_time / 60.0)

    # 최대 우주선 개수 제한
        expected_crs = min(expected_crs, 50)
        num_cosmic_rays = np.random.poisson(expected_crs)

    # 형태별 분포 (60% track, 30% spot, 10% worm)
        morphology_weights = ['track'] * 6 + ['spot'] * 3 + ['worm'] * 1

        for _ in range(num_cosmic_rays):
        # 에너지 (전자 수)
            cr_energy = np.random.lognormal(np.log(15000), 0.8)
            cr_energy = np.clip(cr_energy, 1000, 65000)

        # 형태 선택
            morphology = random.choice(morphology_weights)

        # 시작 위치 (경계에서 충분히 떨어진 곳)
            margin = 20
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
        """점 형태의 우주선을 추가합니다."""
        spot_size = np.random.uniform(0.5, 1.5)
        y_grid, x_grid = np.mgrid[-2:3, -2:3]
        spot_kernel = np.exp(-((x_grid**2 + y_grid**2) / (2 * spot_size**2)))
        
        y_slice = slice(max(0, y-2), min(self.image_size, y+3))
        x_slice = slice(max(0, x-2), min(self.image_size, x+3))
        
        sy, sx = image[y_slice, x_slice].shape
        image[y_slice, x_slice] += spot_kernel[:sy, :sx] * energy

    def _add_cosmic_ray_track(self, image, start_x, start_y, energy):
        """직선 궤적 형태의 우주선을 추가합니다."""
        track_length = int(np.clip(np.random.exponential(10.0), 3, 50))
        angle = np.random.uniform(0, 2 * np.pi)
        
        for step in range(track_length):
            x_pos = int(start_x + step * np.cos(angle))
            y_pos = int(start_y + step * np.sin(angle))
            
            if 0 <= x_pos < self.image_size and 0 <= y_pos < self.image_size:
                # 에너지가 궤적 중앙에서 최대가 되도록 분포
                energy_fraction = np.sin(np.pi * step / track_length) if track_length > 0 else 1
                image[y_pos, x_pos] += energy * energy_fraction / (track_length * 0.64 + 1)

    def _add_cosmic_ray_worm(self, image, start_x, start_y, energy):
        """구불구불한 형태의 우주선을 추가합니다."""
        track_length = int(np.clip(np.random.exponential(12.0), 5, 40))
        angle = np.random.uniform(0, 2 * np.pi)
        waviness = np.random.uniform(2, 5)
        frequency = np.random.uniform(0.3, 0.8)
        
        for step in range(track_length):
            # 기본 직선 경로
            base_x = start_x + step * np.cos(angle)
            base_y = start_y + step * np.sin(angle)
            
            # 구불구불한 변위 추가
            offset_x = waviness * np.sin(frequency * step) * (-np.sin(angle))
            offset_y = waviness * np.sin(frequency * step) * np.cos(angle)
            
            x_pos = int(base_x + offset_x)
            y_pos = int(base_y + offset_y)
            
            if 0 <= x_pos < self.image_size and 0 <= y_pos < self.image_size:
                image[y_pos, x_pos] += energy / track_length

    def _add_stars(self, image, num_stars=50, filter_band='r', add_spikes=True):
        """
        이미지에 무작위 별(점광원)을 추가합니다.
        
        Args:
            image (ndarray): 입력 이미지
            num_stars (int): 추가할 별의 개수
            filter_band (str): 관측 필터
            add_spikes (bool): 밝은 별에 회절 스파이크 추가 여부
            
        Returns:
            ndarray: 별이 추가된 이미지
        """
        star_image = image.copy()
        
        # 대기 조건 시뮬레이션
        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.2), 0.4, 1.5)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.5)
        
        for _ in range(num_stars):
            # 별의 등급 (6등급~22등급)
            mag = np.random.uniform(6.0, 22.0)
            amplitude = 10**((25 - mag) / 2.5) * 100
            
            # 별의 위치
            x_pos = np.random.randint(10, self.image_size - 10)
            y_pos = np.random.randint(10, self.image_size - 10)
            
            # 점광원 생성
            size = 5
            star_kernel = np.zeros((2*size+1, 2*size+1))
            star_kernel[size, size] = amplitude
            star_kernel = convolve(star_kernel, psf_kernel, boundary='fill', fill_value=0)
            
            # 밝은 별에 회절 스파이크 추가
            if add_spikes and mag < 10.0:
                spike_kernel = self._create_spike_kernel(size=25, angle_offset=45)
                spike_intensity = amplitude * 0.1
                star_kernel += convolve(star_kernel, spike_kernel, boundary='fill') * spike_intensity
            
            # 이미지에 별 추가
            y_min, y_max = max(0, y_pos - size), min(self.image_size, y_pos + size + 1)
            x_min, x_max = max(0, x_pos - size), min(self.image_size, x_pos + size + 1)
            
            sy, sx = star_image[y_min:y_max, x_min:x_max].shape
            star_image[y_min:y_max, x_min:x_max] += star_kernel[:sy, :sx]
        
        return star_image

    def _create_spike_kernel(self, size=25, num_spikes=4, angle_offset=45, spike_width=0.8):
        """
        밝은 별의 회절 스파이크를 생성하는 커널을 만듭니다.
        
        Args:
            size (int): 커널 크기
            num_spikes (int): 스파이크 개수
            angle_offset (float): 스파이크 각도 오프셋 (도)
            spike_width (float): 스파이크 두께
            
        Returns:
            ndarray: 정규화된 스파이크 커널
        """
        kernel = np.zeros((size, size))
        center = size // 2
        y, x = np.mgrid[-center:center+1, -center:center+1]
        
        for i in range(num_spikes):
            angle = np.deg2rad(i * (180.0 / (num_spikes/2)) + angle_offset)
            # 직선으로부터의 거리 계산
            dist_from_line = np.abs(x * np.cos(angle) + y * np.sin(angle))
            spike = np.exp(-(dist_from_line**2) / (2 * spike_width**2))
            kernel += spike
        
        return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

    def _add_blooming(self, image, saturation_limit=60000, bleed_fraction=0.8):
        """
        CCD의 전하 블루밍 효과를 시뮬레이션합니다.
        
        Args:
            image (ndarray): 입력 이미지
            saturation_limit (float): 포화 한계 (전자 수)
            bleed_fraction (float): 블리드 비율
            
        Returns:
            ndarray: 블루밍이 적용된 이미지
        """
        bloomed_image = image.copy()
        saturated_pixels = bloomed_image > saturation_limit
        
        if not np.any(saturated_pixels):
            return bloomed_image
        
        # 과잉 전하 계산
        excess_charge_map = np.where(saturated_pixels, bloomed_image - saturation_limit, 0)
        bloomed_image[saturated_pixels] = saturation_limit
        
        # 열 방향으로 블리드 처리
        for c in range(self.image_size):
            excess_in_col = excess_charge_map[:, c]
            if not np.any(excess_in_col > 0):
                continue
            
            # 위쪽으로 블리드
            up_bleed = np.cumsum(excess_in_col[::-1])[::-1] * bleed_fraction
            for r in range(self.image_size - 1, -1, -1):
                available_capacity = saturation_limit - bloomed_image[r, c]
                add_charge = min(up_bleed[r], available_capacity)
                bloomed_image[r, c] += add_charge
                if r > 0:
                    up_bleed[r-1] += (up_bleed[r] - add_charge)
            
            # 아래쪽으로 블리드
            down_bleed = np.cumsum(excess_in_col) * bleed_fraction
            for r in range(self.image_size):
                available_capacity = saturation_limit - bloomed_image[r, c]
                add_charge = min(down_bleed[r], available_capacity)
                bloomed_image[r, c] += add_charge
                if r < self.image_size - 1:
                    down_bleed[r+1] += (down_bleed[r] - add_charge)
        
        return bloomed_image

    def generate_image(self, filter_band='r', exposure_time=30, include_stars=True,
                   num_stars=50, add_blooming=False, cosmic_ray_override=None, verbose=True):


        if verbose:
            print(f"\n🚀 LSST {filter_band}-band 관측 시뮬레이션 시작 (노출: {exposure_time}s)")

        # 1단계: 은하 생성
        galaxy_params = self._generate_realistic_galaxy_params(filter_band)
        allowed_keys = ["amplitude", "r_eff", "n", "x_0", "y_0", "ellip", "theta"]
        sersic_params = {k: v for k, v in galaxy_params.items() if k in allowed_keys}
        ideal_image = Sersic2D(**sersic_params)(self.x, self.y)

        # 2단계: PSF
        seeing_arcsec = np.clip(np.random.lognormal(np.log(0.67), 0.2), 0.4, 1.5)
        psf_kernel = Moffat2DKernel(gamma=seeing_arcsec / self.pixel_scale / 2.35, alpha=2.5)
        convolved_image = convolve(ideal_image, psf_kernel, boundary='extend')

        # 3단계: 별 추가
        if include_stars:
            convolved_image = self._add_stars(convolved_image, num_stars, filter_band)

        # 4단계: 노이즈
        sky_counts = self._get_sky_background_counts(filter_band, exposure_time)
        dark_counts = 0.002 * exposure_time
        base_signal = convolved_image + sky_counts + dark_counts

        # 🚨 Poisson lam 제한
        safe_signal = np.clip(np.maximum(base_signal, 0), 0, 1e7)
        image_with_shot_noise = np.random.poisson(safe_signal)

        read_noise = np.random.normal(0, 8.0, (self.image_size, self.image_size))
        noisy_image = image_with_shot_noise + read_noise

        # 5단계: 우주선
        # 5단계: 우주선
        if cosmic_ray_override is not None:
            final_image, cr_count = self._add_lsst_cosmic_rays(noisy_image, exposure_time)
            cr_count = cosmic_ray_override   # 강제로 덮어쓰기
        else:
            final_image, cr_count = self._add_lsst_cosmic_rays(noisy_image, exposure_time)


        # 6단계: 블루밍
        if add_blooming:
            final_image = self._add_blooming(final_image)

        if verbose:
            galaxy_type = galaxy_params.get('type', 'unknown')
            star_info = f", {num_stars}개 별" if include_stars else ""
            bloom_info = ", 블루밍 적용" if add_blooming else ""
            print(f"✅ 시뮬레이션 완료! ({galaxy_type} 은하{star_info}, {cr_count}개 우주선{bloom_info})")

        return final_image


    