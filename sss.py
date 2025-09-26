from 함수모음 import LSSTAdvancedSimulator
import matplotlib.pyplot as plt

def main():
    """
    단일 천문학적 이미지에 Starlink 궤적 추가 (나쁜 조건 테스트).
    """
    # 시뮬레이터 초기화
    sim = LSSTAdvancedSimulator(image_size=512, seed=42)
    
    # 배경 이미지 생성 (나쁜 조건: 우주선 최대 50개, 별 50개)
    background_image = sim.generate_image(
        filter_band='r',
        exposure_time=30,
        include_stars=True,
        num_stars=50,
        add_blooming=False,  # 블루밍 비활성화 (속도 최적화)
        cosmic_ray_override=50,  # 우주선 최대 50개로 고정
        verbose=True
    )
    
    # 시각화
    plt.imshow(background_image, cmap='gray', origin='lower')
    plt.colorbar()
    plt.title(f"LSST 이미지 (r-band개 Starlink 궤적)")
    plt.show()

if __name__ == "__main__":
    main()