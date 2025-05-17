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

cdef extern from "<arm_sve.h>" nogil:
    pass

cdef extern from "<iostream>" namespace "std":
    pass

cdef extern from "<thread>" namespace "std":
    unsigned int hardware_concurrency()

cdef extern from "seq_sve.h":
    float* scan_sve_impl(
        float* A,
        float* B,
        float* C,
        float* hidden_states,
        float* discrete_time_step,
        float* ssm_state,
        float* scan_output,
        int B_size,
        int D_size,
        int L_size,
        int N_size
    )
    size_t get_sve_vector_length() 