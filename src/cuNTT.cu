#include <string>
#include <exception>
#include <cassert>

#include <thrust/gather.h>
#include <thrust/device_ptr.h>

#include <cuNTT/cuNTT.hpp>
#include <cuNTT/common.cuh>
#include <cuNTT/radix2.cuh>
#include <cuNTT/radix4.cuh>
#include <cuNTT/arithmetic.cuh>

struct cuda_runtime_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

constexpr size_t calc_blocks(size_t len) { return (len + cuNTT_IPB - 1) / cuNTT_IPB; }

void cuda_check(cudaError_t status, const char *action, const char *file, int line) {
    std::string err;
    if(status!=cudaSuccess) {
        err += cudaGetErrorString(status);
        // printf("CUDA error occurred: %s\n", cudaGetErrorString(status));
        if(action!=NULL) {
            err += " while running ";
            err += action;
            err += "  (";
            err += file;
            err += ", line ";
            err += line;
            err += ") \n";
            // printf("While running %s   (file %s, line %d)\n", action, file, line);
        }
        throw cuda_runtime_error(err);
    }
}

void cgbn_check(cgbn_error_report_t *report, const char *file, int32_t line) {
    // check for cgbn errors
    if(cgbn_error_report_check(report)) {
        printf("\n");
        printf("CGBN error occurred: %s\n", cgbn_error_string(report));

        if(report->_instance!=0xFFFFFFFF) {
            printf("Error reported by instance %d", report->_instance);
            if(report->_blockIdx.x!=0xFFFFFFFF || report->_threadIdx.x!=0xFFFFFFFF)
                printf(", ");
            if(report->_blockIdx.x!=0xFFFFFFFF)
                printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            if(report->_threadIdx.x!=0xFFFFFFFF)
                printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
            printf("\n");
        }
        else {
            printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
        }
        if(file!=NULL)
            printf("file %s, line %d\n", file, line);
        throw cuda_runtime_error("CGBN error");
    }
}

#include <cgbn/cgbn.cu>

namespace cuNTT {

void to_mpz(mpz_class& r, const device_mem_t &x, uint32_t count) {
    mpz_import(r.get_mpz_t(), count, -1, sizeof(uint32_t), 0, 0, x._limbs);
}

void from_mpz(device_mem_t &x, const mpz_class& s, uint32_t count) {
    assert(mpz_sizeinbase(s.get_mpz_t(), 2) <= count*32);
    
    size_t words;
    mpz_export(x._limbs, &words, -1, sizeof(uint32_t), 0, 0, s.get_mpz_t());
    while(words < count)
        x._limbs[words++] = 0;
}

void permute_bit_reversal_radix2(device_mem_t *x, int N, int log2N) {
    kernel::permute_bit_reversal<2><<<calc_blocks(N), cuNTT_TPB>>>(x, x, N, log2N);
}

void permute_bit_reversal_radix4(device_mem_t *x, int N, int log2N) {
    kernel::permute_bit_reversal<4><<<calc_blocks(N), cuNTT_TPB>>>(x, x, N, log2N);
}

void EltwiseAddMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseAddMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, y, N, modulus);
}

void EltwiseAddMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseAddMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, scalar, N, modulus);
}

void EltwiseSubMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseSubMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, y, N, modulus);
}

void EltwiseSubMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseSubMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, scalar, N, modulus);
}

void EltwiseSubMod(device_mem_t *out,
                   device_mem_t scalar,
                   const device_mem_t * const __restrict__ x,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseSubMod<<<calc_blocks(N), cuNTT_TPB>>>(out, scalar, x, N, modulus);
}

void EltwiseMultMod(device_mem_t *out,
                    const device_mem_t * const __restrict__ x,
                    const device_mem_t * const __restrict__ y,
                    int N,
                    device_mem_t modulus)
{
    kernel::EltwiseMultMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, y, N, modulus);
}

void EltwiseMultMod(device_mem_t *out,
                    const device_mem_t * const __restrict__ x,
                    device_mem_t scalar,
                    int N,
                    device_mem_t modulus)
{
    kernel::EltwiseMultMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, scalar, N, modulus);
}

void EltwiseMontMultMod(device_mem_t *out,
                        const device_mem_t * const __restrict__ x,
                        device_mem_t adjusted_scalar,
                        int N)
{
    kernel::EltwiseMontMultMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, adjusted_scalar, N);
}

void EltwiseDivMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseDivMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, y, N, modulus);
}

void EltwiseInvMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseInvMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, N, modulus);
}

void EltwiseFMAMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseFMAMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, scalar, y, N, modulus);
}

void EltwiseFMAMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ r,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus)
{
    kernel::EltwiseFMAMod<<<calc_blocks(N), cuNTT_TPB>>>(out, x, r, y, N, modulus);
}

void UpdateQuadratic(device_mem_t *out,
                     const device_mem_t * const __restrict__ x,
                     const device_mem_t * const __restrict__ y,
                     const device_mem_t * const __restrict__ z,
                     device_mem_t r,
                     int N,
                     device_mem_t modulus)
{
    kernel::UpdateQuadratic<<<calc_blocks(N), cuNTT_TPB>>>(out, x, y, z, r, N, modulus);
}

void EltwiseBitDecompose(device_mem_t *out[],
                         const device_mem_t * const __restrict__ x,
                         int N,
                         int bits)
{
    kernel::EltwiseBitDecompose<<<calc_blocks(N), cuNTT_TPB>>>(out, x, N, bits);
}

void ntt_init_global(const mpz_class& p) {
    mpz_class two_p = p * 2, four_p = p * 4, eight_p = p * 8;
    device_mem_t device_p, device_2p, device_4p, device_8p;

    from_mpz(device_p,  p,       cuNTT_LIMBS);
    from_mpz(device_2p, two_p,   cuNTT_LIMBS);
    from_mpz(device_4p, four_p,  cuNTT_LIMBS);
    // from_mpz(device_8p, eight_p, cuNTT_LIMBS);

    CUDA_CHECK(cudaMemcpyToSymbol(gpu_p,  &device_p,  sizeof(device_mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_2p, &device_2p, sizeof(device_mem_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_4p, &device_4p, sizeof(device_mem_t)));
    // CUDA_CHECK(cudaMemcpyToSymbol(gpu_8p, &device_8p, sizeof(device_mem_t)));

    mpz_class J;
    device_mem_t device_J;
    mpz_class beta = mpz_class(1) << cuNTT_BITS;

    mpz_invert(J.get_mpz_t(), p.get_mpz_t(), beta.get_mpz_t());
    from_mpz(device_J, J, cuNTT_LIMBS);
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_J, &device_J, sizeof(device_mem_t)));
}


ntt_context::ntt_context(const mpz_class& p, size_t N, const mpz_class& nth_root_of_unity) : degree_(N) {
    cgbn_error_report_t *err;
    device_mem_t *omega, *omega_inv;
    
    CUDA_CHECK(cgbn_error_report_alloc(&err));
    CUDA_CHECK(cudaMalloc((void **)&omega, sizeof(device_mem_t) * N));
    CUDA_CHECK(cudaMalloc((void **)&omega_inv, sizeof(device_mem_t) * N));

    err_ = std::unique_ptr<cgbn_error_report_t, cgbn_deleter>(err);
    device_omegas_ = std::unique_ptr<device_mem_t, device_deleter>(omega);
    device_omegas_inv_ = std::unique_ptr<device_mem_t, device_deleter>(omega_inv);

    device_mem_t w;
        
    // Precompute Forward NTT omegas
    from_mpz(w, nth_root_of_unity, cuNTT_LIMBS);
    kernel::precompute_omega_table<<<calc_blocks(N), cuNTT_TPB>>>(err_.get(), device_omegas_.get(), w, N);

    // Precompute Inverse NTT omegas
    mpz_class wi, Ni = N;
    mpz_invert(wi.get_mpz_t(), nth_root_of_unity.get_mpz_t(), p.get_mpz_t());
    mpz_invert(Ni.get_mpz_t(), Ni.get_mpz_t(), p.get_mpz_t());
    Ni = (Ni << cuNTT_BITS) % p;    // Adjust for Montgomery multiplication
        
    from_mpz(w, wi, cuNTT_LIMBS);
    from_mpz(N_inv_, Ni, cuNTT_LIMBS);
        
    kernel::precompute_omega_table<<<calc_blocks(N), cuNTT_TPB>>>(err_.get(), device_omegas_inv_.get(), w, N);
}

void ntt_context::ComputeForwardRadix2(device_mem_t *out, device_mem_t * const in) {
    size_t num_blocks = calc_blocks(degree_), num_parallel = 1;

    radix2_fft_forward(err_.get(), out, in,
                       device_omegas_.get(),
                       degree_,
                       dim3(num_blocks, num_parallel),
                       cuNTT_TPB);

}

// NOTE: uncomment 8p to use radix4!
// void ntt_context::ComputeForwardRadix4(device_mem_t *out, device_mem_t * const in) {
//     size_t num_blocks = calc_blocks(degree_), num_parallel = 1;
//     kernel::permute_bit_reversal<4><<<num_blocks, cuNTT_TPB>>> (out, in, degree_, std::log2(degree_));
//     radix4_fft_forward(err_.get(), out, in,
//                        device_omegas_.get(),
//                        degree_,
//                        dim3(num_blocks, num_parallel),
//                        cuNTT_TPB);
// }

void ntt_context::ComputeInverseRadix2(device_mem_t *out, device_mem_t * const in) {
    size_t num_blocks = calc_blocks(degree_), num_parallel = 1;
    radix2_fft_inverse(err_.get(), out, in,
                       device_omegas_inv_.get(),
                       N_inv_,
                       degree_,
                       dim3(num_blocks, num_parallel),
                       cuNTT_TPB);
}

// NOTE: uncomment 8p to use radix4!
// void ntt_context::ComputeInverseRadix4(device_mem_t *out, device_mem_t * const in) {
//     size_t num_blocks = calc_blocks(degree_), num_parallel = 1;
//     radix4_fft_inverse(err_.get(), out, in,
//                        device_omegas_inv_.get(),
//                        N_inv_,
//                        degree_,
//                        dim3(num_blocks, num_parallel),
//                        cuNTT_TPB);
//     kernel::permute_bit_reversal<4><<<num_blocks, cuNTT_TPB>>> (out, out, degree_, std::log2(degree_));
// }

__global__ void gatherKernel(const device_mem_t * __restrict__ input,
                             const size_t * __restrict__ indices,
                             device_mem_t * __restrict__ output,
                             size_t numIndices) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIndices) {
        output[idx] = input[indices[idx]];
    }
}

void sample(device_mem_t *out, const device_mem_t *in, const size_t *index, size_t N)
{
    gatherKernel<<<calc_blocks(N), cuNTT_TPB>>>(in, index, out, N);
}


__global__ void memset_ui_kernel(device_mem_t arr[], int val, size_t N) {
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        arr[idx]._limbs[0] = val;

        #pragma unroll 8
        for (int i = 1; i < cuNTT_LIMBS; i++) {
            arr[idx]._limbs[i] = 0;
        }
    }
}


__global__ void memset_mpz_kernel(device_mem_t arr[], device_mem_t val, size_t N) {
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        arr[idx] = val;
    }
}

void memset_ui(device_mem_t arr[], int val, size_t N) {
    memset_ui_kernel<<<calc_blocks(N), cuNTT_TPB>>>(arr, val, N);
}

void memset_mpz(device_mem_t arr[], device_mem_t val, size_t N) {
    memset_mpz_kernel<<<calc_blocks(N), cuNTT_TPB>>>(arr, val, N);
}

}  // namespace cuNTT

