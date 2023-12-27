#pragma once

namespace cuNTT {

__constant__ device_mem_t gpu_J;
__constant__ device_mem_t gpu_p, gpu_2p, gpu_4p, gpu_8p;

static __constant__ uint32_t reverse_mask_low [5] = {
    0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF
};
static __constant__ uint32_t reverse_mask_high[5] = {
    0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000
};

// Note: Require y = beta * y mod p
__device__ __forceinline__ void
montgomery_mul(env_t& env, num_t& out, const num_t& x, const num_t& y, const num_t& p, const num_t& J) {
    wide_num_t U;

    // U = U1 * beta + U0 (wide product)
    env.mul_wide(U, x, y);

    // Q = U0 * J mod beta (low product)
    env.mul(U._low, U._low, J);

    // H = Q * p / beta (high product)
    env.mul_high(U._low, U._low, p);

    // U0 = U1 - H + p
    env.sub(U._low, p, U._low);
    env.add(out, U._high, U._low);

    // out is in [0, 2p) since we skipped the branching
}

__device__ uint32_t bit_reverse(uint32_t x, uint32_t bits) {
    return __brev(x) >> (32 - bits);
}

template <int radix>
__device__ uint32_t radix_bit_reverse(uint32_t x, uint32_t bits) {
    uint32_t radix_bits = 31 - __clz(radix);
    for (uint32_t i = 4; i >= radix_bits - 1; --i) {
        const uint32_t shift = 1 << i;
        // Recursively swap higher and lower bits
        x = (x & reverse_mask_high[i]) >> shift | (x & reverse_mask_low[i]) << shift;
    }
    return x >> (32 - bits);
}

namespace kernel {

template <size_t radix>
__global__ void
permute_bit_reversal(device_mem_t *out, device_mem_t * const in, int N, int log2N) {
    context_t ctx;
    env_t env(ctx.env<env_t>());

    num_t v, rv;
    for (unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
         tidx < N * cuNTT_TPI;
         tidx += blockDim.x * gridDim.x)
    {
        unsigned int idx = tidx / cuNTT_TPI, ridx;
        if constexpr (radix == 2)
            ridx = bit_reverse(idx, log2N);
        else {
            ridx = radix_bit_reverse<radix>(idx, log2N);
        }
        
        if (idx < ridx) {
            env.load(v, &in[idx]);
            env.load(rv, &in[ridx]);

            env.store(&out[idx], rv);
            env.store(&out[ridx], v);
        }
    }
}


template <size_t input_factor>
__global__ void
reduce_mod(report_error_t *err, device_mem_t * input, int N) {
    static_assert(input_factor == 2 || input_factor == 4 || input_factor == 8);

    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());

    num_t x, factor;
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = (blockIdx.y * N) + curr_thread / cuNTT_TPI;

        env.load(x, &input[instance]);

        if constexpr (input_factor >= 8) {
            env.load(factor, &gpu_4p);
            
            if (env.compare(x, factor) >= 0) {
                env.sub(x, x, factor);
            }
        }

        if constexpr (input_factor >= 4) {
            env.load(factor, &gpu_2p);
            
            if (env.compare(x, factor) >= 0) {
                env.sub(x, x, factor);
            }
        }

        env.load(factor, &gpu_p);
        if (env.compare(x, factor) >= 0) {
            env.sub(x, x, factor);
        }

        env.store(&input[instance], x);
    }
}

__global__ void
adjust_inverse(report_error_t *err, device_mem_t *in, device_mem_t gpu_N_inv, int N)
{
    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());
    
    num_t x;
    num_t p, J, N_inv;

    env.load(p, &gpu_p);
    env.load(J, &gpu_J);
    env.load(N_inv, &gpu_N_inv);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = (blockIdx.y * N) + curr_thread / cuNTT_TPI;

        env.load(x, &in[instance]);
        montgomery_mul(env, x, x, N_inv, p, J);

        // Output is in [0, 2p)
        env.store(&in[instance], x);
    }
 }

__global__ void
precompute_omega_table(report_error_t *err,
                       device_mem_t *omega_table,
                       device_mem_t root_of_unity,
                       int N)
{
    context_t ctx(cgbn_report_monitor, err);
    env_t env(ctx.env<env_t>());

    wide_num_t wide;
    num_t x, p, pow;
    env.load(p, &gpu_p);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;

        env.load(x, &root_of_unity);
        env.set_ui32(pow, instance);
        env.modular_power(x, x, pow, p);

        // Instead of multiply by 2^256, directly set x as high bits
        env.set_ui32(wide._low, 0);  // !! important
        env.set(wide._high, x);
        env.rem_wide(x, wide, p);

        env.store(&omega_table[instance], x);
    }    
}

}  // namespace kernel
}  // namespace cuNTT
