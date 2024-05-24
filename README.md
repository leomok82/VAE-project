
---

## Installation Instructions

To set up the environment and install the package, follow these steps:

### Step 1: Create Conda Environment

1. Open the terminal (or Anaconda Prompt for Windows).

2. Run the following command to create a new Conda environment named `wildfire` with Python version 3.8:
   ```sh
   conda create -n wildfire python=3.8
   ```

3. Activate the newly created environment:
   ```sh
   conda activate wildfire
   ```

### Step 2: Install the Package

1. Make sure you are in the directory containing the `setup.py` file.

2. Run the following command to install the package:
   ```sh
   pip install .
   ```

### Step 3: Verify Installation

1. Enter the Python interpreter by typing `python` in the terminal.

2. Try importing your package to ensure it was installed successfully:
   ```python
   import WildfireThomas
   ```

3. If the import completes without errors, your package has been successfully installed and is ready to use.

---
