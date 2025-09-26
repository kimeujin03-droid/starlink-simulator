# --- 코드 실행 및 시각화 ---
from pro1 import LSSTAdvancedSimulator  # 원래 파일에서 import
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # 시뮬레이터 인스턴스 생성
    sim = LSSTAdvancedSimulator(image_size=512, seed=2025)  # 변수명을 sim으로 변경
    
    # r-band 이미지 생성 (기본 파라미터 사용)
    final_lsst_image = sim.generate_image(filter_band='r', exposure_time=30)
    
    # 결과 시각화
    plt.figure(figsize=(12, 10))
    
    # vmin, vmax를 조절하여 노이즈 속 은하가 잘 보이도록 설정
    plt.imshow(final_lsst_image, cmap='gray_r', origin='lower',
               vmin=np.percentile(final_lsst_image, 5),
               vmax=np.percentile(final_lsst_image, 99.8))
    
    plt.title("LSST Simulated Image (r-band, 30s exposure)", fontsize=16)
    plt.xlabel("X pixels", fontsize=12)
    plt.ylabel("Y pixels", fontsize=12)
    plt.colorbar(label="Pixel Value (ADU-like)")
    
    # 이미지 통계 정보 출력
    print(f"\n📊 생성된 이미지 통계:")
    print(f"- 이미지 크기: {final_lsst_image.shape}")
    print(f"- 최솟값: {np.min(final_lsst_image):.2f}")
    print(f"- 최댓값: {np.max(final_lsst_image):.2f}")
    print(f"- 평균값: {np.mean(final_lsst_image):.2f}")
    print(f"- 표준편차: {np.std(final_lsst_image):.2f}")
    
    plt.tight_layout()
    plt.show()
    
    # 다른 필터와 비교
    print("다른 필터 이미지 생성 중...")
    g_band_image = sim.generate_image(filter_band='g', exposure_time=30)
    i_band_image = sim.generate_image(filter_band='i', exposure_time=30)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (img, band) in enumerate([(final_lsst_image, 'r'), 
                                       (g_band_image, 'g'), 
                                       (i_band_image, 'i')]):
        axes[idx].imshow(img, cmap='gray_r', origin='lower',
                        vmin=np.percentile(img, 5),
                        vmax=np.percentile(img, 99.8))
        axes[idx].set_title(f'{band}-band', fontsize=14)
        axes[idx].set_xlabel('X pixels')
        axes[idx].set_ylabel('Y pixels')
    
    plt.tight_layout()
    plt.show()