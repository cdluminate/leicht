/* lucs.hpp for leicht project
 * Copyright (C) 2017-2018 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
///////////////////////////////////////////////////////////////////////////////
//
// lucs: Lumin's computation subroutines
//
///////////////////////////////////////////////////////////////////////////////
//
// Principle:
//  1. use flattened array forever.
//
// lucs -> BLAS-1 : level 1, vector operation
// lucs -> BLAS-2 : level 2, matrix vector operation
// lucs -> BLAS-3 : level 3, matrix matrix operation
// lucs -> NN     : neural network operations
// TODO: support negative increment?
//
// Reference:
//  1. netlib blas
//
///////////////////////////////////////////////////////////////////////////////

#if !defined(_LEICHT_LUCS_HPP)
#define _LEICHT_LUCS_HPP

#include <cmath>

#if defined(USE_BLAS)
#include <cblas-openblas.h>
#endif

#if defined(USE_OPENMP) && !defined(__clang__)
#include <omp.h>
#elif defined(USE_OPENMP) && defined(__clang__)
#include "/usr/lib/gcc/x86_64-linux-gnu/7/include/omp.h" // FIXME: dirty hack
#endif

namespace lucs {

// BLAS-1 function: ASUM
template <typename Dtype> Dtype
asum(size_t n, Dtype* x, int incx)
{
		Dtype ret = 0.;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret)
#endif
		for (size_t i = 0; i < n; i++) {
			ret += x[i*incx] > (Dtype)0. ? x[i*incx] : -x[i*incx];
		}
		return ret;
}

// BLAS-1 routine: AXPY: Y <- aX + Y
template <typename Dtype> void
axpy(size_t n, Dtype alpha, Dtype* x, int incx, Dtype* y, int incy)
{
#if defined(USE_OPENMP)
#pragma omp parallel for shared(n,alpha,x,y,incx,incy)
#endif
	for (size_t i = 0; i < n; i++)
		y[i*incy] += alpha * x[i*incx];
}

// BLAS-1 routine: COPY: Y <- X
template <typename Dtype> void
copy(size_t n, Dtype* x, int incx, Dtype* y, int incy)
{
	if (incx == 1 && incy == 1) {
		memcpy(y, x, n*sizeof(Dtype)); // Contiguous memory: Very Fast
	} else {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(n,x,incx,y,incy)
#endif
		for (size_t i = 0; i < n; i++)
			y[i*incy] = x[i*incx];
	}
}

// BLAS-1 function: DOT : scalar <- X^T \cdot Y
template <typename Dtype> Dtype
dot(size_t n, Dtype* x, int incx, Dtype* y, int incy)
{
	Dtype ret = 0.;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction(+:ret) shared(n,x,incx,y,incy)
#endif
	for (size_t i = 0; i < n; i++) {
		ret += x[i*incx] * y[i*incy];
	}
	return ret;
}

// BLAS-1 function: NRM2
// XXX: different from the reference BLAS
template <typename Dtype> Dtype
nrm2(size_t n, Dtype* x, int incx)
{
	Dtype ret = 0.;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction(+:ret) shared(n,x,incx)
#endif
	for (size_t i = 0; i < n; i++) {
		ret += x[i*incx] * x[i*incx];
	}
	return std::sqrt(ret);
}

// BLAS-1 routine: SCAL
template <typename Dtype> void
scal(size_t n, Dtype alpha, Dtype* x, int incx) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(n,alpha,x,incx)
#endif
		for (size_t i = 0; i < n; i++)
			x[i*incx] *= alpha;
}

// BLAS-1 function: AMAX
// BLAS-1 function: AMIN
// BLAS-1 routine: SWAP
// BLAS-1 routine: ROT
// BLAS-2 routine: GEMV

// BLAS-3 routine: GEMM: C <- aAB + bC
// Optimized for less Cache Misses, far better than a Naive GEMM impl.
// reference: Netlib Lapack/Blas
#if defined(USE_BLAS)
void
gemm(bool transA, bool transB, size_t M, size_t N, size_t K,
		double alpha, double* A, int lda, double* B, int ldb,
		double beta, double* C, int ldc) {
	cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
			transB ? CblasTrans : CblasNoTrans, M, N, K,
			alpha, A, lda, B, ldb, beta, C, ldc);
}
void
gemm(bool transA, bool transB, size_t M, size_t N, size_t K,
		float alpha, float* A, int lda, float* B, int ldb,
		float beta, float* C, int ldc) {
	cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
			transB ? CblasTrans : CblasNoTrans, M, N, K,
			alpha, A, lda, B, ldb, beta, C, ldc);
}
#else // USE_BLAS
template <typename Dtype> void
gemm(bool transA, bool transB, size_t M, size_t N, size_t K,
		Dtype alpha, Dtype* A, int lda, Dtype* B, int ldb,
		Dtype beta, Dtype* C, int ldc)
{
	if (!transA && !transB) { // A * B
#if defined(USE_OPENMP)
#pragma omp parallel for
#endif // USE_OPENMP
		for (size_t i = 0; i < M; i++) {
			if (beta != 1.) for (size_t j = 0; j < N; j++)
				C[i*ldc+j] *= beta;
			for (size_t k = 0; k < K; k++) {
				Dtype temp = alpha * A[i*lda+k];
#if defined(USE_OPENMP)
#pragma omp simd
#endif // USE_OPENMP
				for (size_t j = 0; j < N; j++)
					C[i*ldc+j] += temp * B[k*ldb+j];
			}
		}
	} else if (transA && !transB) { // A^T * B
#if defined(USE_OPENMP)
#pragma omp parallel for
#endif // USE_OPENMP
		for (size_t i = 0; i < M; i++) {
			if (beta != 1.) for (size_t j = 0; j < N; j++)
				C[i*ldc+j] *= beta;
			for (size_t k = 0; k < K; k++) {
				Dtype temp = alpha * A[k*lda+i];
#if defined(USE_OPENMP)
#pragma omp simd
#endif // USE_OPENMP
				for (size_t j = 0; j < N; j++)
					C[i*ldc+j] += temp * B[k*ldb+j];
			}
		}
	} else if (!transA && transB) { // A * B^T
#if defined(USE_OPENMP)
#pragma omp parallel for collapse(2)
#endif // USE_OPENMP
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j < N; j++) {
				Dtype temp = 0.;
#if defined(USE_OPENMP)
#pragma omp simd
#endif // USE_OPENMP
				for (size_t k = 0; k < K; k++) {
					temp += A[i*lda+k] * B[j*ldb+k];
				}
				C[i*ldc+j] = alpha * temp + beta * C[i*ldc+j];
			}
		}
	} else { // A^T * B^T
#if defined(USE_OPENMP)
#pragma omp parallel for collapse(2)
#endif // USE_OPENMP
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j < N; j++) {
				Dtype temp = 0.;
#if defined(USE_OPENMP)
#pragma omp simd
#endif // USE_OPENMP
				for (size_t k = 0; k < K; k++) {
					temp += A[k*lda+i] * B[j*ldb+k];
				}
				C[i*ldc+j] = alpha * temp + beta * C[i*ldc+j];
			}
		}
	}
}
#endif // USE_BLAS
}
#endif //!defined(_LEICHT_LUCS_HPP)
