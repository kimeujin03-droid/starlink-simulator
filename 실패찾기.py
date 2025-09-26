from 함수모음 import LSSTAdvancedSimulator
import matplotlib.pyplot as plt

# main.py 또는 디버깅용 스크립트에서

# ... 시뮬레이터 생성 ...
sim = LSSTAdvancedSimulator()

# --- 디버깅 시각화 ---
# 1. 이상적인 은하 (노이즈, PSF 없음)
ideal_galaxy = sim._generate_galaxy_component('r')
plt.imshow(ideal_galaxy, cmap='gray_r'); plt.title("1. Ideal Galaxy"); plt.show()

# 2. 이상적인 별 (노이즈, PSF 없음)
ideal_stars = sim._generate_star_component(np.zeros_like(ideal_galaxy), 30.0, 'r')
plt.imshow(ideal_stars, cmap='gray_r'); plt.title("2. Ideal Stars (with Spikes)"); plt.show()

# 3. 은하 + 별 합성
ideal_sky = ideal_galaxy + ideal_stars
plt.imshow(ideal_sky, cmap='gray_r', vmin=0, vmax=1000); plt.title("3. Combined Ideal Sky"); plt.show()

# 4. PSF 적용 후
convolved_sky = sim._apply_psf(ideal_sky)
plt.imshow(convolved_sky, cmap='gray_r', vmin=0, vmax=1000); plt.title("4. After PSF Convolution"); plt.show()

# 5. 스트릭 추가 후
sky_with_streak, _ = sim._add_streak_component(convolved_sky)
plt.imshow(sky_with_streak, cmap='gray_r', vmin=0, vmax=1000); plt.title("5. After Streak Added"); plt.show()

# 6. 최종 노이즈 추가 후
final_image = sim._generate_sky_noise_component(sky_with_streak, 'r', 15.0)
plt.imshow(final_image, cmap='gray_r', vmin=np.percentile(final_image, 1), vmax=np.percentile(final_image, 99)); plt.title("6. Final Noisy Image"); plt.show()