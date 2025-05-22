#Copyright 2025 FUJITSU LIMITED
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# seq_sve.pyx 
# Import necessary modules
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t 
import torch

# Define the data type
DTYPE = np.float32
ctypedef np.float32_t c_dtype

# Declare the external C++ function
cdef extern from "seq_sve.h": 
    float* scan_sve_impl(float* A, float* B, float* C, float* hidden_states, float* discrete_time_step, float* ssm_state, float* scan_output, int B_size, int D_size, int L_size, int N_size)
    size_t get_sve_vector_length() 


def check_vector_length():
    return get_sve_vector_length()

def scan_sve(
    uintptr_t  A, #uintptr_t A,
    uintptr_t  B, #uintptr_t B,
    uintptr_t  C, #uintptr_t C,
    uintptr_t  hidden_states,
    uintptr_t  discrete_time_step,
    uintptr_t  ssm_state,
    uintptr_t  scan_output,
    int B_size, int D_size, int L_size, int N_size
): #-> np.ndarray:
    cdef float* A_ptr = <float*>A
    cdef float* B_ptr = <float*>B
    cdef float* C_ptr = <float*>C 
    cdef float* hidden_states_ptr = <float*>hidden_states
    cdef float* discrete_time_step_ptr = <float*>discrete_time_step
    cdef float* ssm_state_ptr = <float*>ssm_state #<float*>ssm_state
    cdef float* scan_output_ptr = <float*>scan_output
    
    scan_sve_impl(A_ptr, B_ptr, C_ptr, hidden_states_ptr, discrete_time_step_ptr, ssm_state_ptr, scan_output_ptr, B_size, D_size, L_size, N_size)
