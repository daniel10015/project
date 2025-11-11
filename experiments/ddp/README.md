# Setup

**Please** change if there are any issues with the current setup on SOL.

Your environment name on Sol should be "cu13cupti".
If not, change the .sh script accordingly.

## Installation on SOL
0. Start env interactive and load mamba latest:

    `interactive -p htc -c 4 -t 30`

    `module load mamba/latest`

1. Create environment

    `mamba create -n py39cupti -c conda-forge python=3.9`

2. Activate environment

    `source activate py39cupti` 
    - NOTE: You should see `(py39cupti)` to the left of your directory

3. Install dependencies

    `python3.9 -m pip install cupti-python==13.0.0 torch==2.8.0 torchvision==0.23.0`