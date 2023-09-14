#pragma once

#include <gmp.h>
#include <gmpxx.h>
#include <cuda.h>
#include <cgbn/cgbn.h>

inline constexpr size_t TPI  = 4;
inline constexpr size_t TPB  = 256;
inline constexpr size_t BITS = 256;

void cuda_check(cudaError_t status,
                const char *action = nullptr,
                const char *file = nullptr,
                int line = 0);

void cgbn_check(cgbn_error_report_t *report,
                const char *file = nullptr,
                int line = 0);

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

using report_error_t = cgbn_error_report_t;
using device_mem_t = cgbn_mem_t<BITS>;
using context_t = cgbn_context_t<TPI>;
using env_t = cgbn_env_t<context_t, BITS>;
using num_t = typename env_t::cgbn_t;
using wide_num_t = typename env_t::cgbn_wide_t;

namespace cuNTT {

void to_mpz(mpz_class& r, const device_mem_t &x, uint32_t count);
void from_mpz(device_mem_t &x, const mpz_class& s, uint32_t count);

void permute_bit_reversal_radix2(device_mem_t *x, int N, int log2N);
void permute_bit_reversal_radix4(device_mem_t *x, int N, int log2N);

void EltwiseAddMod(device_mem_t *out,
                   device_mem_t * const __restrict__ x,
                   device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

void EltwiseAddMod(device_mem_t *out,
                   device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   int N,
                   device_mem_t modulus);

void EltwiseSubMod(device_mem_t *out,
                   device_mem_t * const __restrict__ x,
                   device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

void EltwiseSubMod(device_mem_t *out,
                   device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   int N,
                   device_mem_t modulus);

void EltwiseMultMod(device_mem_t *out,
                    device_mem_t * const __restrict__ x,
                    device_mem_t * const __restrict__ y,
                    int N,
                    device_mem_t modulus);

void EltwiseMultMod(device_mem_t *out,
                    device_mem_t * const __restrict__ x,
                    device_mem_t scalar,
                    int N,
                    device_mem_t modulus);

void EltwiseFMAMod(device_mem_t *out,
                   device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

/**
   Note: **MUST** call before calling any other GPU NTT functions
   Set p, 2p, 4p, 8p and montgomery factor J on device constant memory
**/
void ntt_init_global(const mpz_class& p);

struct ntt_context {
    ntt_context() = default;
    ntt_context(const mpz_class& p, size_t N, const mpz_class& nth_root_of_unity);

    ~ntt_context(); 

    ntt_context(const ntt_context&) = delete;
    ntt_context& operator=(const ntt_context&) = delete;

    ntt_context(ntt_context&&) = default;
    ntt_context& operator=(ntt_context&&) = default;

    void ComputeForwardRadix2(device_mem_t *out,
                              device_mem_t * const in,
                              int num_blocks,
                              int num_parallel,
                              int threads_per_block);

    void ComputeForwardRadix4(device_mem_t *out,
                              device_mem_t * const in,
                              int num_blocks,
                              int num_parallel,
                              int threads_per_block);

    void ComputeInverseRadix2(device_mem_t *out,
                              device_mem_t * const in,
                              int num_blocks,
                              int num_parallel,
                              int threads_per_block);

    void ComputeInverseRadix4(device_mem_t *out,
                              device_mem_t * const in,
                              int num_blocks,
                              int num_parallel,
                              int threads_per_block);

protected:
    size_t degree_;
    cgbn_error_report_t *err_ = nullptr;
    device_mem_t *device_omegas_ = nullptr;
    device_mem_t *device_omegas_inv_ = nullptr;
    device_mem_t N_inv_;
};


} // namespace cuNTT;
