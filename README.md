# corgihowfsc
A high order wavefront sensing and control simulation suite for the Nancy Grace Roman Space Telescope Coronagraphic Instrument

## Installation Instructions

### Prerequisites

- **Conda/Miniconda** installed on your system
- **Git** installed on your system
- **Internet connection** (for downloading packages and git repos)

### Installation Steps

#### 1. Download Required Files

Download these files manually by clicking on the following links and save them to any **directory of your choice**, e.g.: `/path/to/your/downloads/`:

| File | Download Link                                                                                             |
|------|-----------------------------------------------------------------------------------------------------------|
| `proper_v3.3.3_python.zip` | [Proper Library](https://sourceforge.net/projects/proper-library/files/proper_v3.3.3_python.zip/download) |
| `roman_preflight_proper_public_v2.0.1_python.zip` | [CGISim](https://sourceforge.net/projects/cgisim/files/roman_preflight_proper_public_v2.0.1_python.zip/download)                                                        |
| `cgisim_v4.0.zip` | [CGISim](https://sourceforge.net/projects/cgisim/files/cgisim_v4.0.zip/download)                                                        |

#### 3. Run Installation

Execute these commands in order:

```bash
# Navigate to your corgihowfsc repository clone
cd /path/to/corgihowfsc

# Step 1: Create conda environment and install git-based packages
conda env create -f environment.yml

# Step 2: Activate the environment
conda activate corgiloop

# Step 3: Install manual packages by providing the path to your downloads
python setup_cgi_packages.py /path/to/your/downloads/

# Examples:
python setup_cgi_packages.py ~/Downloads/
python setup_cgi_packages.py C:\Users\username\Downloads\
python setup_cgi_packages.py /home/user/cgi-files/
```

#### 4. Verify Installation

Test that everything is installed correctly:

```python
# In Python, try importing the packages
import proper
import roman_preflight_proper
import cgisim
import howfsc
import eetc

print("✅ All CGI packages imported successfully!")
```

#### Help
```bash
python setup_cgi_packages.py --help
```

### Troubleshooting

#### Missing Downloads
- Verify all 3 zip files are in your specified directory
- Check file names match exactly (case-sensitive)
- Use absolute paths to avoid confusion

#### Environment Issues
- Ensure you activated the environment: `conda activate corgiloop`
- Check conda environment exists: `conda env list`

#### Path Issues
- Use absolute paths: `/full/path/to/downloads/` instead of `~/Downloads/`
- On Windows, use forward slashes or escape backslashes: `C:/Users/name/Downloads/`
- Check directory exists and contains the zip files

#### Installation Failures
- Check internet connection for git repositories
- Verify zip files are not corrupted
- Try running `setup_cgi_packages.py` again
