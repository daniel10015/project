# project
Don't really have a concise name for this yet.
## Dependencies of the project
Cupti-python is only available on Linux, and therefore you must be on a linux distribution to run the project. 
If you're on Windows, then you must run the project on WSL2. You can confirm your WSL version by executing `wsl -l -v` in the command prompt. If you have WSL1, execute `wsl --set-version <Distro> 2` to upgrade to WSL2. Replace `<Distro>` with the distribution you installed WSL with.

### Installation of the project on WSL2

Ensure you have a python version >= python3.9 by doing `python --version`

Create a virtual environment by doing 
```
$ python -m venv [name_of_your_virtual_environment]
```
Click on the link to download the [latest cuda toolkit](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2). 

Execute the `nvidia-smi` command and ensure that the output in both Command Prompt and WSL2 is the same/similar. If nvidia-smi outputs an error similar to the one in NOTE 2, then follow those instructions to fix the issue. Then re-install the toolkit again.

The known version that can run the project is pytorch 12.8, thus, we will install pytorch 12.8 and cupti-python 12.8.

The following code block will show you how to enable the virtual environment and the command to install the packages: 
```
$ source [name_of_your_virtual_environment]/bin/activate
$ pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 cupti-python==12.8
```

Run the python script if you installed everything correctly in 
```
$ proto/verify_torch_version.py
```

NOTE 1: You may need to do some tricky things with versioning to get them all to line up with the driver version.

NOTE 2: If `nvidia-smi` outputs `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.` then run these commands to delete all old instances of cuda toolkit in WSL2. One of our developers had this issue and this solved it.

1. `sudo apt-get remove --purge '^nvidia-.*'`
2. `sudo apt-get remove --purge '^libnvidia-.*'`
3. `sudo apt-get remove --purge '^cuda-.*'`

Run the sample to see if you've installed CUPTI and torch correctly by following [the tutorial](proto/tutorial.md)

## CUPTI tips
- [Link to python tutorial](https://docs.nvidia.com/cupti-python/13.0.0/user-guide/topics/tutorial.html)
