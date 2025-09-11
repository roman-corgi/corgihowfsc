#!/usr/bin/env python3
"""
FITS Data Analysis Script for Modulated Images and Electric Field Estimations

This script processes FITS files containing modulated images and electric field
estimations, creating visualization plots organized by wavelength and iteration.
"""
import glob
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import re


def extract_iteration_number(iteration_dir):
    """Extract iteration number from directory name."""
    match = re.search(r'iteration_(\d+)', iteration_dir)
    return int(match.group(1)) if match else 0


def determine_wavelengths(images_cube, user_wavelengths):
    """
    Determine wavelengths based on user input and validate against cube structure.

    The cube should contain (1 unmodulated + 6 modulated) * num_wavelengths images.

    Args:
        images_cube: The loaded FITS image cube
        user_wavelengths: List of wavelengths in nanometers provided by user

    Returns: List of wavelengths in nanometers
    """
    expected_images = len(user_wavelengths) * 7  # 7 images per wavelength
    actual_images = images_cube.shape[0]

    if actual_images != expected_images:
        raise ValueError(
            f"Mismatch between cube structure and wavelengths!\n"
            f"Expected {expected_images} images for {len(user_wavelengths)} wavelengths, "
            f"but found {actual_images} images in cube."
        )

    return user_wavelengths


def parse_images_cube(images_cube, wavelengths):
    """
    Parse the images cube into wavelength sets.

    Each wavelength has:
    - 1 unmodulated image (index 0)
    - 6 modulated images (indices 1-6): 3 pairs of positive/negative probes

    Args:
        images_cube: The loaded FITS image cube
        wavelengths: List of wavelengths in nanometers

    Returns: dict with wavelength as key and dict of image types as values
    """
    parsed_data = {}

    images_per_wavelength = 7  # 1 unmodulated + 6 modulated

    for i, wavelength in enumerate(wavelengths):
        start_idx = i * images_per_wavelength

        # Extract images for this wavelength
        unmodulated = images_cube[start_idx]
        modulated = images_cube[start_idx + 1:start_idx + 7]

        # Organize modulated images into positive/negative probes
        # Pairs: (positive, negative) for three different probes
        mod_pairs = []
        for j in range(0, 6, 2):
            positive = modulated[j]
            negative = modulated[j + 1]
            mod_pairs.append((positive, negative))

        parsed_data[wavelength] = {
            'unmodulated': unmodulated,
            'probe_pairs': mod_pairs
        }

    return parsed_data


def parse_efield_cube(efield_cube, wavelengths):
    """
    Parse the electric field cube into real and imaginary parts per wavelength.

    The cube contains 2 * num_wavelengths arrays:
    - Real part, then imaginary part for each wavelength

    Returns: dict with wavelength as key and dict of real/imag parts as values
    """
    parsed_efield = {}

    for i, wavelength in enumerate(wavelengths):
        real_idx = i * 2
        imag_idx = i * 2 + 1

        parsed_efield[wavelength] = {
            'real': efield_cube[real_idx],
            'imaginary': efield_cube[imag_idx]
        }

    return parsed_efield


def create_wavelength_plot(images_data, efield_data, wavelength, iteration, output_path):
    """
    Create a 4x2 grid plot for a specific wavelength and iteration.

    Grid layout:
    - Columns 1-3: Positive probes (row 1) and negative probes (row 2)
    - Column 4: Real part of E-field (row 1) and imaginary part (row 2)
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Wavelength {wavelength} nm - Iteration {iteration:04d}', fontsize=16)

    # Plot probe pairs in first 3 columns
    for col in range(3):
        if col < len(images_data['probe_pairs']):
            positive, negative = images_data['probe_pairs'][col]

            # Positive probe (top row)
            im1 = axes[0, col].imshow(positive, cmap='inferno', norm=LogNorm())
            axes[0, col].set_title(f'Positive probe {col + 1}')
            axes[0, col].axis('off')
            plt.colorbar(im1, ax=axes[0, col], shrink=0.6)

            # Negative prone (bottom row)
            im2 = axes[1, col].imshow(negative, cmap='inferno', norm=LogNorm())
            axes[1, col].set_title(f'Negative probe {col + 1}')
            axes[1, col].axis('off')
            plt.colorbar(im2, ax=axes[1, col], shrink=0.6)
        else:
            # If fewer than 3 probe pairs, hide unused subplots
            axes[0, col].axis('off')
            axes[1, col].axis('off')

    # Plot electric field real part (top right)
    im3 = axes[0, 3].imshow(efield_data['real'], cmap='RdBu_r')
    axes[0, 3].set_title('E-field Real Part')
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], shrink=0.6)

    # Plot electric field imaginary part (bottom right)
    im4 = axes[1, 3].imshow(efield_data['imaginary'], cmap='RdBu_r')
    axes[1, 3].set_title('E-field Imaginary Part')
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis function."""
    # =============================================================================
    # USER CONFIGURATION: Define your wavelengths here
    WAVELENGTHS = [523, 550, 578]  # Wavelengths in nanometers
    # =============================================================================

    data_dir = Path('/Users/ilaginja/data_from_repos/corgiloop/acts1')
    output_dir = data_dir / 'probing_plots'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Find all iteration directories
    iteration_dirs = sorted(glob.glob(str(data_dir / 'iteration_*')))

    if not iteration_dirs:
        print("No iteration directories found!")
        return

    print(f"Found {len(iteration_dirs)} iteration directories")
    print(f"Using wavelengths: {WAVELENGTHS} nm")
    print(f"Expected {len(WAVELENGTHS) * 7} images per cube (7 per wavelength)")

    # Process each iteration
    for iteration_dir in iteration_dirs:
        iteration_path = Path(iteration_dir)
        iteration_num = extract_iteration_number(iteration_dir)

        print(f"Processing {iteration_path.name}...")

        # Paths to FITS files
        images_path = iteration_path / 'images.fits'
        efield_path = iteration_path / 'efield_estimations.fits'

        # Check if both files exist
        if not images_path.exists():
            print(f"  Warning: {images_path} not found, skipping...")
            continue
        if not efield_path.exists():
            print(f"  Warning: {efield_path} not found, skipping...")
            continue

        # Load FITS data
        images_cube = fits.getdata(images_path)
        efield_cube = fits.getdata(efield_path)

        # Validate and get wavelengths
        wavelengths = determine_wavelengths(images_cube, WAVELENGTHS)

        # Parse the data
        images_data = parse_images_cube(images_cube, wavelengths)
        efield_data = parse_efield_cube(efield_cube, wavelengths)

        # Create plots for each wavelength
        for wavelength in wavelengths:
            # Create wavelength directory
            wavelength_dir = output_dir / f'{wavelength}'
            wavelength_dir.mkdir(exist_ok=True)

            # Create plot
            plot_filename = f'iteration_{iteration_num:04d}.png'
            plot_path = wavelength_dir / plot_filename

            create_wavelength_plot(
                images_data[wavelength],
                efield_data[wavelength],
                wavelength,
                iteration_num,
                plot_path
            )

        print(f"  ✓ Processed iteration {iteration_num:04d}")

    print(f"\nAnalysis complete! Plots saved to: {output_dir}")
    print(f"Directory structure:")
    for wavelength_dir in sorted(output_dir.iterdir()):
        if wavelength_dir.is_dir():
            plot_count = len(list(wavelength_dir.glob('*.png')))
            print(f"  {wavelength_dir.name}/ ({plot_count} plots)")


if __name__ == '__main__':
    main()
