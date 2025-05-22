/*******************************************************************************
* Copyright 2025 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <thread>
#include <arm_sve.h>  
#include <omp.h> 


size_t get_sve_vector_length() {
    return svcntw();
}

/* Below function was borrowed from the GitHub repository: 
https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/common.hpp */
svfloat32_t exp_ps_sve(svbool_t& pg, svfloat32_t& src) {
    // Constants
    const auto log2_e = svdup_n_f32(1.4426950409f);
    const auto ln2 = svdup_n_f32(0.6931473921f);
    const auto half_ln2_sq = svdup_n_f32(0.2413862043f);
    const auto not_mask17 = svdup_n_u32(~((1u << 17) - 1));
    const auto one = svdup_n_f32(1.0f);
    const svfloat32_t inactive1 = svdup_n_f32(0.0f);
    const svint32_t inactive2 = svdup_n_s32(0);

    // Algorithm starts here
    svfloat32_t t0 = svmul_f32_m(pg, src, log2_e);  // y = x * log2(e)
    svfloat32_t t1 = svrintm_f32_m(inactive1, pg, t0);         // rount to int (float)
    svint32_t t2 = svcvt_s32_f32_m(inactive2, pg, t1);         // n

    t1 = svsub_f32_m(pg, t0, t1);   // a = y - floor(y)
    t1 = svadd_f32_m(pg, t1, one);  // b = a + 1

    svuint32_t t3 = svlsr_n_u32_m(pg, svreinterpret_u32_f32(t1), 17);  // v = b >> 17 (u32)
    svfloat32_t t4 = svexpa_f32(t3);                                   // c = fexpa(v)
    t4 = svscale_f32_m(pg, t4, t2);                                    // fexpa(v) * 2^(n)

    // and_(t2.d, t1.d, not_mask17.d)
    svfloat32_t t5 = svreinterpret_f32_u32(svand_u32_m(pg, svreinterpret_u32_f32(t1), not_mask17));
    t5 = svsub_f32_m(pg, t1, t5);                // z
    t0 = svmla_f32_m(pg, ln2, t5, half_ln2_sq);  // ln2 + half_ln2_sq * z
    t0 = svmla_f32_m(pg, one, t5, t0);           // 1 + (ln2 * z) + (half_ln2_sq * z * z)
    t0 = svmul_f32_m(pg, t0, t4);                // Final result

    return t0;
}

void scan_sve_impl(float* A, float* B, float* C, float* hidden_states,
        float* discrete_time_step, float* ssm_state, float* scan_output, int B_size, int D_size, int L_size, int N_size) {
    
    unsigned int total_cores = std::thread::hardware_concurrency();
    #pragma omp parallel for schedule(dynamic,int(B_size*D_size/total_cores)) collapse(2)
    for (int i = 0; i < B_size; i++) {
        for (int j = 0; j < D_size; j++) {
            for(int l=0; l < L_size; l++ ) {
                int dts_offset = i*D_size*L_size + j*L_size + l;
                svfloat32_t vdts1 = svdup_n_f32(discrete_time_step[ dts_offset]); //load and broadcast value, d_t_s dims: [B_size, D_size, L_size]
                svfloat32_t vhs1 = svdup_n_f32(hidden_states[ dts_offset]); //load and broadcast value, h_s dims: [B_size, D_size, L_size]
                float r1 = 0.0;
                svfloat32_t r1_vector = svdup_n_f32(0.0f);
                for(int k=0; k < N_size; k += svcntw()) {
                    svbool_t pg = svwhilelt_b32(k, N_size);

                    int B_offset = i*L_size*N_size + l*N_size + k;
                    int ssmstate_offset = i*D_size*N_size + j*N_size + k;

                    svfloat32_t vA1 = svld1_f32(pg, &A[ j*N_size + k]); //load A values,  A dims: [D_size, N_size]
                    svfloat32_t vB1 = svld1_f32(pg, &B[ B_offset]); //load B values, B dims: [B_size, L_size, N_size]
                    svfloat32_t vC1 = svld1_f32(pg, &C[ B_offset]); //load C values, C dims: [B_size, L_size, N_size]
                    svfloat32_t vssm1 = svld1_f32(pg, &ssm_state[ ssmstate_offset]);
                    svfloat32_t t1 = svmul_f32_m(pg, vdts1,vA1); // perform vds1*vA1 & vds1*vA2 which forms discrete_A
                    t1 = exp_ps_sve(pg, t1);
                    svfloat32_t t3 = svmul_f32_m(pg, vdts1,vB1); //t3 & t4 contains first 8 chunks and last 8 chunks of discrete_B
                    svfloat32_t t5 = svmul_f32_m(pg, vhs1,t3);  // perform vhs1*t3 and vhs1*t4 which forms deltaB_u

                    // Sequential scan algorithm starts here
                    vssm1 = svmad_f32_m(pg, vssm1, t1, t5); //perform discrete_A*ssm_state + deltaB_u in two chunks 
                    r1_vector = svadd_f32_m(pg, svmul_f32_m(pg, vssm1, vC1), r1_vector);  // perform matmul(ssm_state, C) in two chunks
                    svst1_f32(pg, &ssm_state[ ssmstate_offset], vssm1);
                }
                scan_output[ dts_offset] += svaddv(svptrue_b32(), r1_vector);  // scan_output dims: [B_size, D_size, L_size]
            }
        }
    }
}
