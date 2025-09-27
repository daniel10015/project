# project
Don't really have a concise name for this yet.
## Steps to download cupti-python
Ensure you have a python version`>= python3.9 by doing python --version

Create a virtual environment by doing python -m venv [name_of_your_virtual_environment]`

If you're on Windows, then ensure you have WSL2 by running wsl -l -v in command prompt. 
Then follow the cuda toolkit: https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2
Ensure that nvidia-smi in the WSL2 terminal has the same/similar output in the Command Prompt.

Enable your virtual environment and execute this command.  
Check the driver version of your GPU and ensure your CUPTI and pytorch versions match up
1. `pip install torch`cupti-python==[version_of_the_downloaded_toolkit]

NOTE: You may need to do some tricky things with versioning to get them all to line up with the driver version.

Run the sample to see if you've installed CUPTI and torch correctly by following [the tutorial](proto/tutorial.md)

## CUPTI tips
- [Link to python tutorial](https://docs.nvidia.com/cupti-python/13.0.0/user-guide/topics/tutorial.html)
