# TODO tests for demangling

# The 1st kernel is what's used for a linear layer
# the 2nd kernel is what's used for relu
# the 3rd kernel is softmax (turning logits into probability distribution)
mangled_list = ['_Z17gemv2T_kernel_valIiiffffLi128ELi16ELi4ELi4ELb0ELb1E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_T4_S7_',
                '_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_49_GLOBAL__N__d2ba64fb_16_TensorCompare_cu_71e06f4e19launch_clamp_scalarERNS_18TensorIteratorBaseEN3c106ScalarES6_NS0_6detail11ClampLimitsEENKUlvE_clEvENKUlvE5_clEvEUlfE_St5arrayIPcLm2EEEEviT0_T1_',
                '_ZN43_GLOBAL__N__b6de9c8c_10_SoftMax_cu_9f978f6320softmax_warp_forwardIfffLi4ELb0ELb0EEEvPT0_PKT_iiiPKbib']

def demangle(mangled):
  pass

for mangled in mangled_list:
  demangle(mangled)