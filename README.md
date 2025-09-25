# project
Don't really have a concise name for this yet.
## Steps to download cupti-python
Check the driver version of your GPU and ensure your CUPTI and pytorch versions match up
1. Ensure you have a python version `>= python3.9`
2. `python3.<Version> -m pip install torch`
3. `python3.<Version> -m pip install nvidia-cuda-cupti`
4. `python3.<Version> -m pip install cupti-python`

NOTE: You may need to do some tricky things with versioning to get them all to line up with the driver version.

Errors related to driver versioning most likely means you'll need to downgrade CUPTI, or upgrade torch. In most cases it's preferable to downgrade to match the driver version, which you check by doing `nvcc --version`.

Run the sample to see if you've installed CUPTI and torch correctly by following [the tutorial](proto/tutorial.md)

## CUPTI tips
- [Link to python tutorial](https://docs.nvidia.com/cupti-python/13.0.0/user-guide/topics/tutorial.html)