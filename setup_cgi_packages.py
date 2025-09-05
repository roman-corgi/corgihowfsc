#!/usr/bin/env python3
"""
CGI Manual Packages Setup Script

This script installs the CGI packages that must be downloaded manually.
Make sure you have:
1. Created the conda environment: conda env create -f environment.yml
2. Activated the environment: conda activate corgiloop
3. Downloaded all required files to a directory

Usage:
  python setup_cgi_packages.py /path/to/files/    # Uses absolute path
"""

import os
import subprocess
import sys
from pathlib import Path
import zipfile
import argparse


def check_environment():
    """Check if we're in the correct conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'corgiloop':
        print("❌ ERROR: Please activate the corgiloop environment first:")
        print("   conda activate corgiloop")
        sys.exit(1)
    print(f"✓ Using conda environment: {conda_env}")


def run_pip_install(package_dir):
    """Install a package using pip in the current environment"""
    print(f"   Installing from {package_dir}...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", ".", "-e"],
                                cwd=package_dir,
                                check=True,
                                capture_output=True,
                                text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Installation failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False


def get_downloads_directory():
    """Get the downloads directory from command line or use default"""
    parser = argparse.ArgumentParser(
        description="Install CGI packages from downloaded zip files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_cgi_packages.py /home/user/cgi-files/     # Use absolute path
  python setup_cgi_packages.py C:\\Users\\user\\cgi\\    # Windows example
        """
    )
    parser.add_argument(
        'downloads_path',
        nargs='?',
        default='downloads',
        help='Path to directory containing downloaded zip files'
    )

    args = parser.parse_args()
    downloads_dir = Path(args.downloads_path).resolve()

    return downloads_dir


def main():
    print("=== CGI Manual Packages Installation ===\n")

    # Check environment
    check_environment()

    # Get downloads directory
    downloads_dir = get_downloads_directory()

    # Define package information
    packages = [
        {
            'name': 'Proper',
            'zip': 'proper_v3.3.3_python.zip',
            'dir': 'proper_v3.3.3_python',
            'url': 'https://sourceforge.net/projects/proper-library/'
        },
        {
            'name': 'Roman Preflight Proper',
            'zip': 'roman_preflight_proper_public_v2.0.1_python.zip',
            'dir': 'roman_preflight_proper_public_v2.0.1_python',
            'url': 'https://sourceforge.net/projects/cgisim/'
        },
        {
            'name': 'CGISim',
            'zip': 'cgisim_v4.0.zip',
            'dir': 'cgisim_v4.0',
            'url': 'https://sourceforge.net/projects/cgisim/'
        }
    ]

    # Set up directories
    work_dir = Path("cgi_extracted")
    work_dir.mkdir(exist_ok=True)

    # Validate downloads directory
    if not downloads_dir.exists():
        print(f"❌ ERROR: Downloads directory not found: {downloads_dir}")
        print("Please provide a valid path to the directory containing the zip files.")
        print("\nExample usage:")
        print("  python setup_cgi_packages.py /path/to/your/downloads/")
        sys.exit(1)

    print(f"📁 Downloads directory: {downloads_dir}")
    print(f"📁 Working directory: {work_dir.absolute()}\n")

    # Check for all required files first
    missing_files = []
    for pkg in packages:
        zip_path = downloads_dir / pkg['zip']
        if not zip_path.exists():
            missing_files.append({'zip': pkg['zip'], 'url': pkg['url']})

    if missing_files:
        print("❌ ERROR: Missing required files in downloads directory:")
        for missing in missing_files:
            print(f"   • {missing['zip']}")
            print(f"     Download from: {missing['url']}")
        print(f"\nPlease download all files to: {downloads_dir}")
        sys.exit(1)

    print("✓ All required zip files found!\n")

    # Process each package
    success_count = 0
    for i, pkg in enumerate(packages, 1):
        print(f"[{i}/{len(packages)}] Processing {pkg['name']}...")

        zip_path = downloads_dir / pkg['zip']
        dir_path = work_dir / pkg['dir']

        # Extract if needed
        if not dir_path.exists():
            print(f"   📦 Extracting {pkg['zip']}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(work_dir)
                print(f"   ✓ Extracted to {dir_path}")
            except Exception as e:
                print(f"   ❌ Extraction failed: {e}")
                continue
        else:
            print(f"   📁 Directory already exists: {dir_path}")

        # Install package
        if run_pip_install(dir_path):
            print(f"   ✅ {pkg['name']} installed successfully\n")
            success_count += 1
        else:
            print(f"   ❌ {pkg['name']} installation failed\n")

    # Summary
    print("=" * 50)
    if success_count == len(packages):
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("\nYour CGI environment is ready to use!")
        print("The corgiloop environment is already activated.")
    else:
        print(f"⚠️  {success_count}/{len(packages)} packages installed successfully")
        print("Please check the errors above and try again.")

    print("\n📝 Note: Extracted files are in cgi_extracted/ (you can delete this later)")
    return success_count == len(packages)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Installation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)