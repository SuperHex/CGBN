#pragma once

namespace cuNTT::kernel {

__global__ void
EltwiseAddMod(device_mem_t *out,
              device_mem_t * const __restrict__ x,
              device_mem_t * const __restrict__ y,
              int N,
              device_mem_t modulus)
{
    context_t ctx(cgbn_report_monitor);
    env_t env(ctx.env<env_t>());

    num_t a, b, p;
    env.load(b, &modulus);
    
    for (int curr_thread = blockDim.x * blockIdx.x + threadIdx.x;
         curr_thread < N * cuNTT_TPI;
         curr_thread += blockDim.x * gridDim.x)
    {
        const int instance = curr_thread / cuNTT_TPI;
        
        env.load(a, &x[instance]);
        env.load(b, &y[instance]);
        
        env.add(a, a, b);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseAddMod(device_mem_t *out,
              device_mem_t * const __restrict__ x,
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
        
        env.load(a, &x[instance]);
        env.add(a, a, s);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseSubMod(device_mem_t *out,
              device_mem_t * const __restrict__ x,
              device_mem_t * const __restrict__ y,
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
        
        env.load(a, &x[instance]);
        env.load(b, &y[instance]);
        
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
              device_mem_t * const __restrict__ x,
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
        
        env.load(a, &x[instance]);
        
        env.add(a, a, p);
        env.sub(a, a, s);
        
        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseMultMod(device_mem_t *out,
               device_mem_t * const __restrict__ x,
               device_mem_t * const __restrict__ y,
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
        
        env.load(a, &x[instance]);
        env.load(b, &y[instance]);
        
        env.mul_wide(tmp, a, b);
        env.rem_wide(a, tmp, p);

        env.store(&out[instance], a);
    }
}

__global__ void
EltwiseMultMod(device_mem_t *out,
               device_mem_t * const __restrict__ x,
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
        
        env.load(a, &x[instance]);

        env.mul_wide(tmp, a, s);
        env.rem_wide(a, tmp, p);

        env.store(&out[instance], a);
    }
}


__global__ void
EltwiseFMAMod(device_mem_t *out,
              device_mem_t * const __restrict__ x,
              device_mem_t scalar,
              device_mem_t * const __restrict__ y,
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
        
        env.load(a, &x[instance]);
        env.load(b, &y[instance]);

        env.mul_wide(tmp, a, s);
        env.rem_wide(a, tmp, p);

        env.add(a, a, b);

        if (env.compare(a, p) >= 0) {
            env.sub(a, a, p);
        }

        env.store(&out[instance], a);
    }
}

}  // namespace cuNTT::kernel
