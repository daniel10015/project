#!/bin/bash

# Runs the sample scripts to ensure correct build and setup.
cd proto


python3 verify_torch_version.py
echo
echo
echo
python3 sample_matmul.py
echo
echo
echo
python3 sample_model.py