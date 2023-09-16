#pragma once

namespace cuNTT {

void radix2_fft_forward(report_error_t * err,
                        device_mem_t * out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_table,
                        int N,
                        dim3 block_dim,
                        int threads_per_block);

void radix2_fft_inverse(report_error_t * err,
                        device_mem_t * out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_inv_table,
                        device_mem_t N_inv,
                        int N,
                        dim3 block_dim,
                        int threads_per_block);

namespace kernel {

__global__ void
radix2_DIF_butterfly(report_error_t *err,
                     device_mem_t *out,
                     device_mem_t * const __restrict__ in,
                     device_mem_t * const __restrict__ w_precomputed_table,
                     int N,
                     int iteration)
{
    const int M = 1 << iteration;
    const int M2 = M >> 1;
    
    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());
    
    num_t J;
    env.load(J, &gpu_J);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI / 2;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;   // instance is the true `i` for [0, N/2)
        const int omega_stride = N / M;
        const int fft_group = instance / M2;
        const int fft_M_index = instance % M2;
        const int k = (blockIdx.y * N) + (fft_group * M + fft_M_index);

        num_t x, y, tmp;

        env.load(x, &in[k]);
        env.load(y, &in[k + M2]);

        env.add(tmp, x, y);
        env.sub(y, x, y);

        env.load(x, &gpu_2p);
        int borrow = env.sub(tmp, tmp, x);
        if (borrow) {
            env.add(tmp, tmp, x);
        }
        env.store(&out[k], tmp);

        // ========== X is done ==========
        env.add(y, y, x);
        
        env.load(x, &gpu_p);
        env.load(tmp, &w_precomputed_table[fft_M_index * omega_stride]);

        montgomery_mul(env, y, y, tmp, x, J);
        env.store(&out[k + M2], y);
    }
}



__global__ void
radix2_DIT_butterfly(report_error_t *err,
                     device_mem_t *out,
                     device_mem_t * const __restrict__ in,
                     device_mem_t * const __restrict__ w_precomputed_table,
                     int N,
                     int iteration)
{
    const int M = 1 << iteration;
    const int M2 = M >> 1;
    
    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());

    num_t J;
    env.load(J, &gpu_J);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI / 2;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        const int omega_stride = N / M;
        const int fft_group = instance / M2;
        const int fft_M_index = instance % M2;
        const int k = (blockIdx.y * N) + (fft_group * M + fft_M_index);

        num_t x, y, tmp;

        env.load(x, &gpu_p);
        env.load(tmp, &w_precomputed_table[fft_M_index * omega_stride]);
        env.load(y, &in[k + M2]);
        
        montgomery_mul(env, y, y, tmp, x, J);

        env.load(x, &in[k]);
        env.load(tmp, &gpu_2p);

        int borrow = env.sub(x, x, tmp);
        if (borrow) {
            env.add(x, x, tmp);
        }

        env.sub(tmp, tmp, y);  // y = y - 2p
        env.add(tmp, x, tmp);
        env.store(&out[k + M2], tmp);

        env.add(tmp, x, y);
        env.store(&out[k], tmp);
    }
}

} // namespace kernel

void radix2_fft_forward(report_error_t *err,
                        device_mem_t *out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_table,
                        int N,
                        dim3 block_dim,
                        int threads_per_block)
{
    const int iterations = std::log2(N);

    // dim3 block_size(num_blocks, num_parallel);

    kernel::radix2_DIF_butterfly<<<block_dim, threads_per_block>>>(err, out, in, omega_table, N, iterations);
    
    for (int iter = iterations - 1; iter >= 1; --iter) {
        kernel::radix2_DIF_butterfly<<<block_dim, threads_per_block>>>(err, out, out, omega_table, N, iter);
    }

    kernel::reduce_mod<2><<<block_dim, threads_per_block>>>(err, out, N);
}

void radix2_fft_inverse(report_error_t *err,
                        device_mem_t *out,
                        device_mem_t * const __restrict__ in,
                        device_mem_t * const __restrict__ omega_inv_table,
                        device_mem_t N_inv,
                        int N,
                        dim3 block_dim,
                        int threads_per_block)
{
    const int iterations = std::log2(N);

    // dim3 block_size(num_blocks, num_parallel);
    kernel::radix2_DIT_butterfly<<<block_dim, threads_per_block>>>(err, out, in, omega_inv_table, N, 1);
    
    for (int iter = 2; iter <= iterations; ++iter) {
        kernel::radix2_DIT_butterfly<<<block_dim, threads_per_block>>>(err, out, out, omega_inv_table, N, iter);
    }
    kernel::adjust_inverse<<<block_dim, threads_per_block>>>(err, out, N_inv, N);
    kernel::reduce_mod<2><<<block_dim, threads_per_block>>>(err, out, N);
}

}  // namespace cuNTT
