import numpy as np
import matplotlib.pyplot as plt
from starlink_simulator import LSSTAdvancedSimulator

def main():
    """
    Main function to demonstrate the LSSTAdvancedSimulator.
    Generates and displays a single realistic astronomical image with a Starlink streak.
    """
    print("🚀 Starting LSST Advanced Simulator to generate an image...")

    # 1. Initialize the simulator.
    # Change the 'seed' value for different galaxy, star, and noise patterns.
    simulator = LSSTAdvancedSimulator(image_size=512, seed=2024)

    # 2. Generate the final image including a satellite streak.
    # The generate_image function handles all complex internal processes.
    sim_image = simulator.generate_image(
        filter_band='r',
        exposure_time=30,
        include_satellites=True,
        satellite_probability=1.0 # 100% probability to ensure a streak is generated.
    )

    # 3. Visualize the generated image.
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
    main()