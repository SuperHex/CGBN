/*
 * sha256.cu Implementation of SHA256 Hashing    
 *
 * Date: 12 June 2019
 * Revision: 1
 * *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

 
/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>
#include <stdint.h>
#include <cuNTT/cuNTT.hpp>

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/



/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
__constant__ uint32_t k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DEFINITIONS ***********************/
namespace cuSHA {

__device__  __forceinline__ void
cuda_sha256_perm(sha256_cuda_ctx *ctx, int idx) {
	uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (ctx->data[j][idx] << 24)
             | (ctx->data[j + 1][idx] << 16)
             | (ctx->data[j + 2][idx] << 8)
             | (ctx->data[j + 3][idx]);
    
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0][idx];
	b = ctx->state[1][idx];
	c = ctx->state[2][idx];
	d = ctx->state[3][idx];
	e = ctx->state[4][idx];
	f = ctx->state[5][idx];
	g = ctx->state[6][idx];
	h = ctx->state[7][idx];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[0][idx] += a;
	ctx->state[1][idx] += b;
	ctx->state[2][idx] += c;
	ctx->state[3][idx] += d;
	ctx->state[4][idx] += e;
	ctx->state[5][idx] += f;
	ctx->state[6][idx] += g;
	ctx->state[7][idx] += h;
}

__device__ void cuda_sha256_final(sha256_cuda_ctx *ctx, uint8_t hash[], int idx) {
	uint32_t i;

	i = ctx->datalen[idx];

	// Pad whatever data is left in the buffer.
	if (ctx->datalen[idx] < 56) {
		ctx->data[i++][idx] = 0x80;
		while (i < 56)
			ctx->data[i++][idx] = 0x00;
	}
	else {
		ctx->data[i++][idx] = 0x80;
		while (i < 64)
			ctx->data[i++][idx] = 0x00;
		cuda_sha256_perm(ctx, idx);
		// memset(ctx->data, 0, 56);

        for (int i = 0; i < 56; i++)
            ctx->data[i][idx] = 0;
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen[idx] += ctx->datalen[idx] * 8;

    uint64_t bitlen = ctx->bitlen[idx];
	ctx->data[63][idx] = bitlen;
	ctx->data[62][idx] = bitlen >> 8;
	ctx->data[61][idx] = bitlen >> 16;
	ctx->data[60][idx] = bitlen >> 24;
	ctx->data[59][idx] = bitlen >> 32;
	ctx->data[58][idx] = bitlen >> 40;
	ctx->data[57][idx] = bitlen >> 48;
	ctx->data[56][idx] = bitlen >> 56;
	cuda_sha256_perm(ctx, idx);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
    uint32_t base = idx * SHA256_BLOCK_SIZE;
    for (i = 0; i < 8; i++) {
        const uint32_t state = ctx->state[i][idx];
        const int offset = base + i * 4;
        hash[offset]     = (state >> 24) & 0x000000ff;
        hash[offset + 1] = (state >> 16) & 0x000000ff;
        hash[offset + 2] = (state >> 8)  & 0x000000ff;
        hash[offset + 3] = (state)       & 0x000000ff;
    }
}

__global__ void
kernel_sha256_init(sha256_cuda_ctx *ctx, size_t N) {
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        ctx->datalen[idx]  = 0;
        ctx->bitlen[idx]   = 0;
        ctx->state[0][idx] = 0x6a09e667;
        ctx->state[1][idx] = 0xbb67ae85;
        ctx->state[2][idx] = 0x3c6ef372;
        ctx->state[3][idx] = 0xa54ff53a;
        ctx->state[4][idx] = 0x510e527f;
        ctx->state[5][idx] = 0x9b05688c;
        ctx->state[6][idx] = 0x1f83d9ab;
        ctx->state[7][idx] = 0x5be0cd19;
    }
}

__global__ void
kernel_sha256_update(sha256_cuda_ctx *ctx,
                     const uint8_t *data,
                     size_t num_elements,
                     size_t per_element_bytes)
{
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < num_elements;
         idx += blockDim.x * gridDim.x)
    {
        for (int i = 0; i < per_element_bytes; ++i) {
            int datalen = ctx->datalen[idx];
            ctx->data[datalen][idx] = data[idx * per_element_bytes + i];
            ctx->datalen[idx]++;
            if (ctx->datalen[idx] == 64) {
                cuda_sha256_perm(ctx, idx);
                ctx->bitlen[idx]  += 512;
                ctx->datalen[idx] = 0;
            }
        }
    }
}

__global__ void
kernel_sha256_final(sha256_cuda_ctx *ctx, uint8_t *out, size_t N) {
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        cuda_sha256_final(ctx, out, idx);
    }
}

constexpr size_t cuSHA_TPB = 512;
constexpr size_t cuSHA_calc_grid(size_t N) { return (N + cuSHA_TPB - 1) / cuSHA_TPB; }

void sha256_malloc_member(sha256_cuda_ctx *ctx, size_t N) {
    uint8_t* data[64];
    for (int i = 0; i < 64; i++) {
        cudaMalloc(&data[i], N * sizeof(uint8_t));
    }
    cudaMemcpy(&(ctx->data), data, 64 * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    
    uint32_t *datalen;
    cudaMalloc(&datalen, N * sizeof(uint32_t));
    cudaMemcpy(&(ctx->datalen), &datalen, sizeof(uint32_t*), cudaMemcpyHostToDevice);

    uint64_t *bitlen;
    cudaMalloc(&bitlen, N * sizeof(uint64_t));
    cudaMemcpy(&(ctx->bitlen), &bitlen, sizeof(uint64_t*), cudaMemcpyHostToDevice);

    uint32_t* state[8];
    for (int i = 0; i < 8; i++) {
        cudaMalloc(&state[i], N * sizeof(uint32_t));
    }
    cudaMemcpy(&(ctx->state), state, 8 * sizeof(uint32_t*), cudaMemcpyHostToDevice);
}

void sha256_free_member(sha256_cuda_ctx *ctx) {
    // Temporary host array to hold device pointers for data
    uint8_t* data[64];
    // Copy the array of device pointers from device to host
    cudaMemcpy(data, ctx->data, 64 * sizeof(uint8_t*), cudaMemcpyDeviceToHost);
    // Free each individual array
    for (int i = 0; i < 64; i++) {
        cudaFree(data[i]);
    }
    
    // Free datalen
    uint32_t* datalen;
    // Copy the pointer from device to host
    cudaMemcpy(&datalen, &(ctx->datalen), sizeof(uint32_t*), cudaMemcpyDeviceToHost);
    cudaFree(datalen);
    
    // Free bitlen
    uint64_t* bitlen;
    // Copy the pointer from device to host
    cudaMemcpy(&bitlen, &(ctx->bitlen), sizeof(uint64_t*), cudaMemcpyDeviceToHost);
    cudaFree(bitlen);

    // Temporary host array to hold device pointers for state
    uint32_t* state[8];
    // Copy the array of device pointers from device to host
    cudaMemcpy(state, ctx->state, 8 * sizeof(uint32_t*), cudaMemcpyDeviceToHost);
    // Free each individual array
    for (int i = 0; i < 8; i++) {
        cudaFree(state[i]);
    }   
}

void sha256_init(sha256_cuda_ctx *ctx, size_t N) {
    kernel_sha256_init<<<cuSHA_calc_grid(N), cuSHA_TPB>>>(ctx, N);
}

void sha256_update(sha256_cuda_ctx *ctx, const uint8_t *data, size_t N, size_t batch_size) {
    kernel_sha256_update<<<cuSHA_calc_grid(N), cuSHA_TPB>>>(ctx, data, N, batch_size);
}

void sha256_update(sha256_cuda_ctx *ctx, const cuNTT::device_mem_t *data, size_t N) {
    kernel_sha256_update<<<cuSHA_calc_grid(N), cuSHA_TPB>>>(
        ctx,
        reinterpret_cast<const uint8_t*>(data),
        N,
        cuNTT_LIMBS * sizeof(uint32_t));
}

void sha256_final(sha256_cuda_ctx *ctx, uint8_t *out, size_t N) {
    kernel_sha256_final<<<cuSHA_calc_grid(N), cuSHA_TPB>>>(ctx, out, N);
}

}  // namespace cuSHA

