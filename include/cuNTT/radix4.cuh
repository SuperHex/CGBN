#pragma once

namespace cuNTT {

void radix4_fft_forward(report_error_t * err,
                        device_mem_t * out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_table,
                        int N,
                        dim3 block_dim,
                        int threads_per_block);

void radix4_fft_inverse(report_error_t * err,
                        device_mem_t * out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_inv_table,
                        device_mem_t N_inv,
                        int N,
                        dim3 block_dim,
                        int threads_per_block);

namespace kernel {

__global__ void
radix4_DIF_butterfly(report_error_t *err,
                     device_mem_t *out,
                     device_mem_t * const __restrict__ in,
                     device_mem_t * const __restrict__ w_precomputed_table,
                     int N,
                     int iteration)
{
    const int M = 1 << (iteration << 1);
    const int M2 = M >> 1;
    const int M4 = M >> 2;
    const int M34 = M2 + M4;

    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());

    num_t p, four_p, eight_p, J;
    env.load(p, &gpu_p);
    env.load(four_p, &gpu_4p);
    env.load(eight_p, &gpu_8p);
    env.load(J, &gpu_J);    

    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * TPI / 4;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / TPI;   // instance is the true `i` for [0, N/4)
        const int omega_stride = N / M;
        const int fft_group = instance / M4;
        const int fft_M_index = instance % M4;
        const int k = (blockIdx.y * N) + (fft_group * M + fft_M_index);

        // Assumption: A, B, C, D in [0, 4p)
        num_t A, B, C, D;
        num_t wk, w2k, w3k, w4;

        env.load(A, &in[k]);
        env.load(B, &in[k + M4]);
        env.load(C, &in[k + M2]);
        env.load(D, &in[k + M34]);
        
        env.load(wk,  &w_precomputed_table[fft_M_index * omega_stride]);
        env.load(w2k, &w_precomputed_table[fft_M_index * 2 * omega_stride]);
        env.load(w3k, &w_precomputed_table[fft_M_index * 3 * omega_stride]);
        env.load(w4,  &w_precomputed_table[N / 4]);

        // x = u + v mod 2p
        num_t a0, a1, b0, b1;
        env.add(a0, A, C);

        env.add(a1, A, four_p);
        env.sub(a1, a1, C);
        
        env.add(b0, B, D);

        env.add(b1, B, four_p);
        env.sub(b1, b1, D);

        montgomery_mul(env, b1, b1, w4, p, J);

        num_t u, v, s, t;
        env.add(u, a0, b0);
        
        env.add(v, a1, b1);
        montgomery_mul(env, v, v, wk, p, J);

        env.add(s, a0, eight_p);
        env.sub(s, s, b0);
        montgomery_mul(env, s, s, w2k, p, J);

        env.add(t, a1, eight_p);
        env.sub(t, t, b1);
        montgomery_mul(env, t, t, w3k, p, J);

        if (env.compare(u, eight_p) >= 0) {
            env.sub(u, u, eight_p);
        }

        if (env.compare(u, four_p) >= 0) {
            env.sub(u, u, four_p);
        }

        env.store(&out[k],       u);
        env.store(&out[k + M4],  v);
        env.store(&out[k + M2],  s);
        env.store(&out[k + M34], t);
    }
}

__global__ void
radix4_DIT_butterfly(report_error_t *err,
                     device_mem_t *out,
                     device_mem_t * const __restrict__ in,
                     device_mem_t * const __restrict__ w_precomputed_table,
                     int N,
                     int iteration)
{
    const int M = 1 << (iteration << 1);
    const int M2 = M >> 1;
    const int M4 = M >> 2;
    const int M34 = M2 + M4;

    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());

    num_t p, two_p, four_p, J;
    env.load(p, &gpu_p);
    env.load(two_p, &gpu_2p);
    env.load(four_p, &gpu_4p);
    env.load(J, &gpu_J);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * TPI / 4;
         curr_thread += blockDim.x * gridDim.x)
    {
        // const int thread_pos = threadIdx.x + blockDim.x * blockIdx.x;
        const int instance = curr_thread / TPI;
        const int omega_stride = N / M;
        const int fft_group = instance / M4;
        const int fft_M_index = instance % M4;
        const int k = (blockIdx.y * N) + (fft_group * M + fft_M_index);

        // Assumption: A, B, C, D in [0, 8p)
        num_t A, B, C, D;
        num_t wk, w2k, wkN4;

        env.load(A, &in[k]);
        env.load(B, &in[k + M4]);
        env.load(C, &in[k + M2]);
        env.load(D, &in[k + M34]);
        
        env.load(wk,    &w_precomputed_table[fft_M_index * omega_stride]);
        env.load(w2k,   &w_precomputed_table[fft_M_index * 2 * omega_stride]);
        env.load(wkN4,  &w_precomputed_table[fft_M_index * omega_stride + N / 4]);

        // Reduce A to [0, 4p)
        if (env.compare(A, four_p) >= 0) {
            env.sub(A, A, four_p);
        }

        num_t a0, a1, b0, b1;
        {
            num_t tmp;
            montgomery_mul(env, tmp, w2k, C, p, J);
        
            env.add(a0, A, tmp);

            env.add(a1, A, two_p);
            env.sub(a1, a1, tmp);
            // At this point a0, a1 in [0, 4p+2p)
        }
        
        {
            num_t tmp;
            montgomery_mul(env, tmp, w2k, D, p, J);

            env.add(b0, B, tmp);
            montgomery_mul(env, b0, wk, b0, p, J);

            env.add(b1, B, two_p);
            env.sub(b1, b1, tmp);
            montgomery_mul(env, b1, wkN4, b1, p, J);

            // At this point b0, b1 in [0, 2p)
        }

        num_t u, v, s, t;
        env.add(u, a0, b0);
        env.add(v, a1, b1);

        env.add(s, a0, two_p);
        env.sub(s, s, b0);

        env.add(t, a1, two_p);
        env.sub(t, t, b1);

        env.store(&out[k],       u);
        env.store(&out[k + M4],  v);
        env.store(&out[k + M2],  s);
        env.store(&out[k + M34], t);
    }
}

}  // namespace kernel

void radix4_fft_forward(report_error_t *err,
                        device_mem_t *out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_table,
                        int N,
                        dim3 block_dim,
                        int threads_per_block)
{
    const int iterations = std::log2(N) / 2;

    kernel::radix4_DIF_butterfly<<<block_dim, threads_per_block>>>(err, out, in, omega_table, N, iterations);
    
    for (int iter = iterations - 1; iter >= 1; --iter) {
        kernel::radix4_DIF_butterfly<<<block_dim, threads_per_block>>>(err, out, out, omega_table, N, iter);
    }

    kernel::reduce_mod<4><<<block_dim, threads_per_block>>>(err, out, N);
}

void radix4_fft_inverse(report_error_t *err,
                        device_mem_t *out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_inv_table,
                        device_mem_t N_inv,
                        int N,
                        dim3 block_dim,
                        int threads_per_block)
{
    const int iterations = std::log2(N) / 2;

    kernel::radix4_DIT_butterfly<<<block_dim, threads_per_block>>>(err, out, in, omega_inv_table, N, 1);
    
    for (int iter = 2; iter <= iterations; ++iter) {
        kernel::radix4_DIT_butterfly<<<block_dim, threads_per_block>>>(err, out, out, omega_inv_table, N, iter);
    }
    kernel::adjust_inverse<<<block_dim, threads_per_block>>>(err, out, N_inv, N);
    kernel::reduce_mod<2><<<block_dim, threads_per_block>>>(err, out, N);
}

}  // namespace cuNTT
