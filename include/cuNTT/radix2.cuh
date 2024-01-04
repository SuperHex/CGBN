#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace cuNTT {

void radix2_fft_forward(report_error_t * err,
                        device_mem_t * out,
                        device_mem_t * const in,
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
                     device_mem_t * const in,
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

__global__ void
radix2_DIF_butterfly_small(report_error_t *err,
                           device_mem_t * out,
                           device_mem_t * const in,
                           device_mem_t * const __restrict__ w_precomputed_table,
                           int N,
                           int iteration)
{
    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());
    
    num_t J;
    env.load(J, &gpu_J);

    num_t p, pp;
    env.load(p,  &gpu_p);
    env.load(pp, &gpu_2p);

    // ------------------------------------------------------------
    extern __shared__ device_mem_t cache[];

    const unsigned int instance = threadIdx.x / cuNTT_TPI;
    const unsigned int block_size = blockDim.x >> 1;
    const unsigned int half_block_size = block_size >> 1;
    const unsigned int global_base = blockIdx.x * block_size;

    // Load into cache for current block
    {
        num_t a, b;
        
        env.load(a, &in[global_base + instance]);
        env.load(b, &in[global_base + instance + half_block_size]);
        
        env.store(&cache[instance], a);
        env.store(&cache[instance + half_block_size], b);


        __syncthreads();
    }

    // ------------------------------------------------------------

    for (int iter = iteration; iter >= 1; iter--) {
        const int M = 1 << iter;
        const int M2 = M >> 1;
        const int omega_stride = N / M;

        num_t x, y, u, v;
    
        // const int instance = idx / cuNTT_TPI;
        const int fft_group = instance / M2;
        const int fft_M_index = instance % M2;
        const int k = fft_group * M + fft_M_index;

        env.load(x, &cache[k]);
        env.load(y, &cache[k + M2]);

        env.add(u, x, y);
        env.sub(v, x, y);

        int borrow = env.sub(u, u, pp);
        if (borrow) {
            env.add(u, u, pp);
        }
        env.store(&cache[k], u);

        env.add(v, v, pp);
        
        env.load(x, &w_precomputed_table[fft_M_index * omega_stride]);

        montgomery_mul(env, v, v, x, p, J);
        env.store(&cache[k + M2], v);

        __syncthreads();
    }


    // Store the result back to global memory
    {
        num_t a, b;
        
        env.load(a, &cache[instance]);
        env.load(b, &cache[instance + half_block_size]);

        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        if (env.compare(b, p) >= 0) {
            env.sub(b, b, p);
        }

        env.store(&out[global_base + instance], a);
        env.store(&out[global_base + instance + half_block_size], b);
    }
}


__global__ void
radix2_DIT_butterfly_small(report_error_t *err,
                           device_mem_t *out,
                           device_mem_t * const in,
                           device_mem_t * const __restrict__ w_precomputed_table,
                           int N,
                           int iteration)
{
        
    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());

    num_t J;
    env.load(J, &gpu_J);

    // ------------------------------------------------------------
    extern __shared__ device_mem_t cache[];

    const unsigned int instance = threadIdx.x / cuNTT_TPI;
    const unsigned int block_size = blockDim.x >> 1;
    const unsigned int half_block_size = block_size >> 1;
    const unsigned int global_base = blockIdx.x * block_size;

    {
        num_t a, b;
        
        env.load(a, &in[global_base + instance]);
        env.load(b, &in[global_base + instance + half_block_size]);
        
        env.store(&cache[instance], a);
        env.store(&cache[instance + half_block_size], b);


        __syncthreads();
    }

    // ------------------------------------------------------------
        
    for (int iter = 1; iter <= iteration; iter++) {
        const int M = 1 << iter;
        const int M2 = M >> 1;
        const int omega_stride = N / M;
        
        const int fft_group = instance / M2;
        const int fft_M_index = instance % M2;
        const int k = (blockIdx.y * N) + (fft_group * M + fft_M_index);

        num_t x, y, tmp;

        env.load(x, &gpu_p);
        env.load(tmp, &w_precomputed_table[fft_M_index * omega_stride]);
        env.load(y, &cache[k + M2]);
        
        montgomery_mul(env, y, y, tmp, x, J);

        env.load(x, &cache[k]);
        env.load(tmp, &gpu_2p);

        int borrow = env.sub(x, x, tmp);
        if (borrow) {
            env.add(x, x, tmp);
        }

        env.sub(tmp, tmp, y);  // y = y - 2p
        env.add(tmp, x, tmp);
        env.store(&cache[k + M2], tmp);

        env.add(tmp, x, y);
        env.store(&cache[k], tmp);

        __syncthreads();
    }

    {
        num_t a, b;
        
        env.load(a, &cache[instance]);
        env.load(b, &cache[instance + half_block_size]);

        env.store(&out[global_base + instance], a);
        env.store(&out[global_base + instance + half_block_size], b);
    }
}

} // namespace kernel

// void radix2_fft_forward(report_error_t *err,
//                         device_mem_t *out,
//                         device_mem_t * in,
//                         device_mem_t * omega_table,
//                         int N,
//                         dim3 block_dim,
//                         int threads_per_block)
// {
//     int iterations = std::log2(N);

//     kernel::radix2_DIF_butterfly<<<block_dim, threads_per_block>>>(err, out, in, omega_table, N, iterations);
    
//     for (int iter = iterations - 1; iter >= 1; --iter) {
//         kernel::radix2_DIF_butterfly<<<block_dim, threads_per_block>>>(err, out, out, omega_table, N, iter);
//     }

//     kernel::reduce_mod<2><<<block_dim, threads_per_block>>>(err, out, N);
// }

void radix2_fft_forward(report_error_t *err,
                        device_mem_t * out,
                        device_mem_t * in,
                        device_mem_t * omega_table,
                        int N,
                        dim3 block_dim,
                        int threads_per_block)
{
    const int log2N = std::log2(N);

    constexpr int small_log2 = 8;
    constexpr int small_size = 256;
    constexpr int small_threads = small_size * cuNTT_TPI / 2;
    const int small_blocks = (N + small_size - 1) / small_size;

    kernel::permute_bit_reversal<2><<<block_dim, threads_per_block>>> (out, in, N, log2N);

    int iter;
    for (iter = log2N; iter > small_log2; --iter) {
        kernel::radix2_DIF_butterfly<<<block_dim, threads_per_block>>>(err, out, out, omega_table, N, iter);
    }

    // result reduced in small kernel
    kernel::radix2_DIF_butterfly_small<<<small_blocks, small_threads, small_size * sizeof(device_mem_t)>>>(err, out, out, omega_table, N, iter);
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
    const int log2N = std::log2(N);

    constexpr int small_log2 = 8;
    constexpr int small_size = 256;
    constexpr int small_threads = small_size * cuNTT_TPI / 2;
    const int small_blocks = (N + small_size - 1) / small_size;

    kernel::radix2_DIT_butterfly_small<<<small_blocks, small_threads, small_size * sizeof(device_mem_t)>>>(err, out, in, omega_inv_table, N, small_log2);
    
    for (int iter = small_log2 + 1; iter <= log2N; ++iter) {
        kernel::radix2_DIT_butterfly<<<block_dim, threads_per_block>>>(err, out, out, omega_inv_table, N, iter);
    }
    
    kernel::adjust_inverse_reduce<<<block_dim, threads_per_block>>>(err, out, N_inv, N);
    kernel::permute_bit_reversal<2><<<block_dim, threads_per_block>>> (out, out, N, log2N);
}

}  // namespace cuNTT
