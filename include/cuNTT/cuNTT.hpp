#pragma once

#include <memory>
#include <gmp.h>
#include <gmpxx.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cgbn/cgbn.h>

inline constexpr size_t cuNTT_TPI  = 4;                     // threads per instance
inline constexpr size_t cuNTT_TPB  = 256;                   // threads per block
inline constexpr size_t cuNTT_IPB = cuNTT_TPB / cuNTT_TPI;  // instances per block

inline constexpr size_t cuNTT_BITS = 256;
inline constexpr size_t cuNTT_LIMBS = (cuNTT_BITS + 31) / 32;

void cuda_check(cudaError_t status,
                const char *action = nullptr,
                const char *file = nullptr,
                int line = 0);

void cgbn_check(cgbn_error_report_t *report,
                const char *file = nullptr,
                int line = 0);

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

namespace cuNTT {

using report_error_t = cgbn_error_report_t;
using device_mem_t = cgbn_mem_t<cuNTT_BITS>;
using context_t = cgbn_context_t<cuNTT_TPI>;
using env_t = cgbn_env_t<context_t, cuNTT_BITS>;
using num_t = typename env_t::cgbn_t;
using wide_num_t = typename env_t::cgbn_wide_t;

void to_mpz(mpz_class& r, const device_mem_t &x, uint32_t count = cuNTT_LIMBS);
void from_mpz(device_mem_t &x, const mpz_class& s, uint32_t count = cuNTT_LIMBS);

void permute_bit_reversal_radix2(device_mem_t *x, int N, int log2N);
void permute_bit_reversal_radix4(device_mem_t *x, int N, int log2N);

void EltwiseAddMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

void EltwiseAddMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   int N,
                   device_mem_t modulus);

void EltwiseSubMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

void EltwiseSubMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   int N,
                   device_mem_t modulus);

void EltwiseSubMod(device_mem_t *out,
                   device_mem_t scalar,
                   const device_mem_t * const __restrict__ x,
                   int N,
                   device_mem_t modulus);

void EltwiseMultMod(device_mem_t *out,
                    const device_mem_t * const __restrict__ x,
                    const device_mem_t * const __restrict__ y,
                    int N,
                    device_mem_t modulus);

void EltwiseMultMod(device_mem_t *out,
                    const device_mem_t * const __restrict__ x,
                    device_mem_t scalar,
                    int N,
                    device_mem_t modulus);

void EltwiseDivMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

void EltwiseInvMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   int N,
                   device_mem_t modulus);

void EltwiseFMAMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   device_mem_t scalar,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

void EltwiseFMAMod(device_mem_t *out,
                   const device_mem_t * const __restrict__ x,
                   const device_mem_t * const __restrict__ r,
                   const device_mem_t * const __restrict__ y,
                   int N,
                   device_mem_t modulus);

void UpdateQuadratic(device_mem_t *out,
                     const device_mem_t * const __restrict__ x,
                     const device_mem_t * const __restrict__ y,
                     const device_mem_t * const __restrict__ z,
                     device_mem_t r,
                     int N,
                     device_mem_t modulus);

void EltwiseBitDecompose(device_mem_t *out[],
                         const device_mem_t * const __restrict__ x,
                         int N,
                         int bits);

/**
   Note: **MUST** call before calling any other GPU NTT functions
   Set p, 2p, 4p, 8p and montgomery factor J on device constant memory
**/
void ntt_init_global(const mpz_class& p);

struct ntt_context {
    struct cgbn_deleter {
        void operator()(cgbn_error_report_t * p) const {
            CUDA_CHECK(cgbn_error_report_free(p));
        }        
    };
    struct device_deleter {
        void operator()(device_mem_t * p) const {
            CUDA_CHECK(cudaFree(p));
        }
    };
    
    ntt_context() = default;
    ntt_context(const mpz_class& p, size_t N, const mpz_class& nth_root_of_unity);

    void ComputeForwardRadix2(device_mem_t *out,
                              device_mem_t * const in);

    void ComputeForwardRadix4(device_mem_t *out,
                              device_mem_t * const in);

    void ComputeInverseRadix2(device_mem_t *out,
                              device_mem_t * const in);

    void ComputeInverseRadix4(device_mem_t *out,
                              device_mem_t * const in);

protected:
    size_t degree_;
    std::unique_ptr<cgbn_error_report_t, cgbn_deleter> err_          = nullptr;
    std::unique_ptr<device_mem_t, device_deleter> device_omegas_     = nullptr;
    std::unique_ptr<device_mem_t, device_deleter> device_omegas_inv_ = nullptr;
    device_mem_t N_inv_;
};


void sample(device_mem_t *out, const device_mem_t *in, const size_t *index, size_t N);
void memset_ui(device_mem_t arr[], int val, size_t N);
void memset_mpz(device_mem_t arr[], device_mem_t val, size_t N);

} // namespace cuNTT


namespace cuSHA {

struct SHA256_CTX {
	uint8_t data[64];
	uint32_t datalen;
	uint64_t bitlen;
	uint32_t state[8];
};

void sha256_init(SHA256_CTX ctxs[], size_t N);
void sha256_update(SHA256_CTX ctxs[], const uint8_t *data, size_t N, size_t batch_size);
void sha256_update(SHA256_CTX ctxs[], const cuNTT::device_mem_t *data, size_t N);
void sha256_final(SHA256_CTX ctxs[], uint8_t *out, size_t N);

}  // namespace cuSHA
