# Starlink Satellite Streak Simulator

This project is a **Python simulator that generates realistic Starlink satellite streaks on virtual astronomical observation images, as if captured by the LSST (Vera C. Rubin Observatory)**.

Beyond simply drawing lines, it simulates various physical phenomena occurring during actual astronomical observations, producing highly realistic results.

## Key Features

*   **Realistic Astronomical Simulation**: Generates backgrounds similar to real astronomical photos, including galaxies (Sersic profile), background stars, and atmospheric effects (PSF).
*   **Physics-Based Noise Models**: Accurately simulates various noise types originating from CCD sensors, such as shot noise, read noise, sky background brightness, and cosmic rays.
*   **TLE-Based Satellite Trajectory Calculation**: Uses actual satellite orbital data (TLE) to precisely calculate where a satellite will pass in the sky at a given observation time.
*   **High-Quality Streak Rendering**: Based on the calculated trajectory, it renders high-quality streaks considering factors like satellite distance, brightness, thickness, and blooming effects from very bright light sources.

## How to Run
You can run the `starlink_simulator.py` file directly to see the simulation results.
```bash
python main.py
```
