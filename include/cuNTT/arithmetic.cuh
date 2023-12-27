#pragma once

namespace cuNTT::kernel {

__global__ void
EltwiseAddMod(device_mem_t *out,
              device_mem_t const * const __restrict__ x,
              device_mem_t const * const __restrict__ y,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    const env_t env(ctx.env<env_t>());

    num_t a, b, p;
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;

        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.load(b, const_cast<device_mem_t*>(&y[instance]));
        
        env.add(a, a, b);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseAddMod(device_mem_t *out,
              const device_mem_t * const __restrict__ x,
              device_mem_t scalar,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t a, s, p;
    env.load(s, &scalar);
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.add(a, a, s);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseSubMod(device_mem_t *out,
              const device_mem_t * const __restrict__ x,
              const device_mem_t * const __restrict__ y,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t a, b, p;
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.load(b, const_cast<device_mem_t*>(&y[instance]));
        
        env.add(a, a, p);
        env.sub(a, a, b);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseSubMod(device_mem_t *out,
              const device_mem_t * const __restrict__ x,
              device_mem_t scalar,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t a, s, p;
    env.load(s, &scalar);
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        
        env.add(a, a, p);
        env.sub(a, a, s);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}

__global__ void
EltwiseSubMod(device_mem_t *out,
              device_mem_t scalar,
              const device_mem_t * const __restrict__ x,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t a, s, p;
    env.load(s, &scalar);
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        
        env.add(s, s, p);
        env.sub(a, s, a);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseMultMod(device_mem_t *out,
               const device_mem_t * const __restrict__ x,
               const device_mem_t * const __restrict__ y,
               int N,
               device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    wide_num_t tmp;
    num_t a, b, p;
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.load(b, const_cast<device_mem_t*>(&y[instance]));
        
        env.mul_wide(tmp, a, b);
        env.rem_wide(a, tmp, p);

        env.store(&out[instance], a);
    }
}

__global__ void
EltwiseMultMod(device_mem_t *out,
               const device_mem_t * const __restrict__ x,
               device_mem_t scalar,
               int N,
               device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    wide_num_t tmp;
    num_t a, s, p;
    env.load(s, &scalar);
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));

        env.mul_wide(tmp, a, s);
        env.rem_wide(a, tmp, p);

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseFMAMod(device_mem_t *out,
              const device_mem_t * const __restrict__ x,
              device_mem_t scalar,
              const device_mem_t * const __restrict__ y,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    wide_num_t tmp;
    num_t a, b, s, p;
    env.load(s, &scalar);
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.load(b, const_cast<device_mem_t*>(&y[instance]));

        env.mul_wide(tmp, a, s);
        env.rem_wide(a, tmp, p);

        env.add(a, a, b);

        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}

// Calculate out = r * x + y;
__global__ void
EltwiseFMAMod(device_mem_t *out,
              const device_mem_t * const __restrict__ x,
              const device_mem_t * const __restrict__ r,
              const device_mem_t * const __restrict__ y,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    wide_num_t tmp;
    num_t a, b, R, p;
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;

        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.load(R, const_cast<device_mem_t*>(&r[instance]));
        env.load(b, const_cast<device_mem_t*>(&y[instance]));

        env.mul_wide(tmp, a, R);
        env.rem_wide(a, tmp, p);

        env.add(a, a, b);

        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
UpdateQuadratic(device_mem_t *out,
                const device_mem_t * const __restrict__ x,
                const device_mem_t * const __restrict__ y,
                const device_mem_t * const __restrict__ z,
                device_mem_t r,
                int N,
                device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t p, J;
    num_t a, b, rand;
    wide_num_t wide;

    env.load(rand, &r);
    env.load(p, &modulus);
    env.load(J, &gpu_J);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;

        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.load(b, const_cast<device_mem_t*>(&y[instance]));

        env.mul_wide(wide, a, b);
        env.rem_wide(a, wide, p);

        env.load(b, const_cast<device_mem_t*>(&z[instance]));

        int borrow = env.sub(a, a, b);
        if (borrow) {
            env.add(a, a, p);
        }

        env.mul_wide(wide, a, rand);
        env.rem_wide(a, wide, p);

        // Add result to accumulator

        env.load(b, &out[instance]);

        env.add(a, a, b);

        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}

__global__ void
EltwiseInvMod(device_mem_t *out,
              const device_mem_t * const __restrict__ x,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t a, p;
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.modular_inverse(a, a, p);
        env.store(&out[instance], a);
    }
}

__global__ void
EltwiseDivMod(device_mem_t *out,
              const device_mem_t * const __restrict__ x,
              const device_mem_t * const __restrict__ y,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    wide_num_t wide;
    num_t a, b, p;
    env.load(p, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        env.load(a, const_cast<device_mem_t*>(&x[instance]));
        env.load(b, const_cast<device_mem_t*>(&y[instance]));
        env.modular_inverse(b, b, p);

        env.mul_wide(wide, a, b);
        env.rem_wide(a, wide, p);

        env.store(&out[instance], a);
    }
}

__global__ void
EltwiseBitDecompose(device_mem_t *out[],
                    const device_mem_t * const __restrict__ x,
                    int N,
                    int bits)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t a;
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, const_cast<device_mem_t*>(&x[instance]));

        num_t val, bit;
        for (int k = 0; k < bits; k++) {
            env.shift_right(val, a, k);
            env.bitwise_mask_and(bit, val, 1);
            env.store(&out[k][instance], bit);
        }
    }
}

}  // namespace cuNTT::kernel
