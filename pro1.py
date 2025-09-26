# run_simulation.py
from lsst_simulator import LSSTAdvancedSimulator

# 시뮬레이터 초기화
sim = LSSTAdvancedSimulator(image_size=512, seed=42)

# 원하는 조건으로 최종 한 장 생성
final_image = sim.generate_image(
    filter_band='r',
    exposure_time=30,
    include_stars=True,
    num_stars=50,
    add_blooming=False
)

# 결과 시각화
import matplotlib.pyplot as plt
plt.imshow(final_image, cmap='gray', origin='lower')
plt.colorbar()
plt.title("LSST simulated image (r-band)")
plt.show()