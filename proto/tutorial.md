# Tutorial

Run `sample_matmul.py` and your output should look something like this:
```
PyTorch version: 2.8.0+cu126
CUPTI module loaded: cupti.cupti
kernel name = _ZN2at6native54_GLOBAL__N__d8ceb000_21_DistributionNormal_cu_0c5b6e8543distribution_elementwise_grid_stride_kernelIfLi4EZNS0_9templates4cuda20normal_and_transformIffPNS_17CUDAGeneratorImplEZZZNS4_13normal_kernelIS7_EEvRKNS_10TensorBaseEddT_ENKUlvE_clEvENKUlvE0_clEvEUlfE_EEvRNS_18TensorIteratorBaseET1_T2_EUlP24curandStatePhilox4_32_10E0_ZNS1_27distribution_nullary_kernelIff6float4S7_SM_SF_EEvSH_SJ_RKT3_T4_EUlifE_EEvlNS_15PhiloxCudaStateESI_SJ_
kernel duration (ns) = 39330
kernel name = _ZN2at6native54_GLOBAL__N__d8ceb000_21_DistributionNormal_cu_0c5b6e8543distribution_elementwise_grid_stride_kernelIfLi4EZNS0_9templates4cuda20normal_and_transformIffPNS_17CUDAGeneratorImplEZZZNS4_13normal_kernelIS7_EEvRKNS_10TensorBaseEddT_ENKUlvE_clEvENKUlvE0_clEvEUlfE_EEvRNS_18TensorIteratorBaseET1_T2_EUlP24curandStatePhilox4_32_10E0_ZNS1_27distribution_nullary_kernelIff6float4S7_SM_SF_EEvSH_SJ_RKT3_T4_EUlifE_EEvlNS_15PhiloxCudaStateESI_SJ_
kernel duration (ns) = 39394
kernel name = volta_sgemm_128x64_nn
kernel duration (ns) = 996770
Computation done, profiling captured.
Result tensor shape: torch.Size([1024, 1024])
```

Errors related to driver versioning most likely means you'll need to downgrade CUPTI, or upgrade torch. In most cases it's preferable to downgrade to match the driver version, which you check by doing `nvcc --version`.

You can also run `sample_model.py` which should output something similar to:
```
Memcpy Host -> Device of 1728 bytes on stream 7 duration (ns) = 1440
Memcpy Host -> Device of 64 bytes on stream 7 duration (ns) = 352
Memcpy Host -> Device of 18432 bytes on stream 7 duration (ns) = 1792
Memcpy Host -> Device of 128 bytes on stream 7 duration (ns) = 352
Memcpy Host -> Device of 1280 bytes on stream 7 duration (ns) = 864
Memcpy Host -> Device of 40 bytes on stream 7 duration (ns) = 352
----------Training complete----------
loss: 2.2960925102233887
```