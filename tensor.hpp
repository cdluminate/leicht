/* tensor.cc for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#if !defined(_LEICHT_TENSOR_HPP)
#define _LEICHT_TENSOR_HPP

#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <climits>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <unordered_map> // faster than <map>
#include <vector>

#include <jsoncpp/json/json.h>
#include "leicht.hpp"

#if defined(USE_BLAS)
#include <cblas-openblas.h>
#endif

#if defined(USE_OPENMP) && !defined(__clang__)
#include <omp.h>
#elif defined(USE_OPENMP) && defined(__clang__)
#include "/usr/lib/gcc/x86_64-linux-gnu/7/include/omp.h" // FIXME: dirty hack
#endif

using namespace std;


///////////////////////////////////////////////////////////////////////////////
//
// lLAS: lumin/leicht (Non-Basic) Linear Algebra Subroutines
//
///////////////////////////////////////////////////////////////////////////////
//
// llas -> BLAS-1 : level 1, vector operation
// llas -> BLAS-2 : level 2, matrix vector operation
// llas -> BLAS-3 : level 3, matrix matrix operation
// Reference: Netlib BLAS/LAPACK
// TODO: will the tricks used in reference BLAS help?
// TODO: support negative increment?
//
// begin [[ lLAS ]] -----------------------------------------------------------
namespace llas {

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
//ut                                                                  llas/asum
//>                     cout << endl; Tensor<double> x(5); x.rand_(); x.dump();
//>                         cout << llas::asum(x.getSize(), x.data, 1) << endl;
//ut                                                             llas/asum incx
//>                   cout << endl; Tensor<double> x(3,5); x.rand_(); x.dump();
//>                                   cout << llas::asum(3, x.data, 5) << endl;

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
//ut                                                                  llas/axpy
//>                     cout << endl; Tensor<double> x(5); x.rand_(); x.dump();
//>                                   Tensor<double> y(5); y.rand_(); y.dump();
//>                                    llas::axpy(5, .5, x.data, 1, y.data, 1);
//>                                                                   y.dump();

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
#else
template <typename Dtype> void
gemm(bool transA, bool transB, size_t M, size_t N, size_t K,
		Dtype alpha, Dtype* A, int lda, Dtype* B, int ldb,
		Dtype beta, Dtype* C, int ldc)
{
	if (!transA && !transB) { // A * B
        //#pragma omp parallel for collapse(2)
		//for (size_t i = 0; i < M; i++) {
		//	for (size_t j = 0; j < N; j++) {
		//		Dtype vdot = beta * C[i*ldc+j];
		//		for (size_t k = 0; k < K; k++) {
		//			vdot += alpha * A[i*lda+k] * B[k*ldb+j];
		//		}
		//		C[i*ldc+j] = vdot;
		//	}
		//} // I5-2520M, OMPthread=2, 512x512 double gemm, 10 run, 2384 ms. (Naive version)
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
		} // I5-2520M, OMPthread=2, 512x512 double gemm, 10 run, 553 ms.
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
		} // I5-2520M, OMPthread=2, 512x512 double gemm, 10 run, 629 ms. (Naive version 13295 ms)
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
		} // I5-2520M, OMPthread=2, 512x512 double gemm, 10 run, 710 ms. (= Naive version)
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
		} // I5-2520M, OMPthread=2, 512x512 double gemm, 10 run, 2421 ms. (Naive version 4750 ms)
	}
}
#endif // USE_BLAS
//ut                                                  llas/gemm noTrans noTrans
//>         Tensor<double> x(3,5); x.rand_(); Tensor<double> y(5,2); y.rand_();
//>                                                      Tensor<double> z(3,2);
//> llas::gemm(false, false, 3, 2, 5, 1., x.data, 5, y.data, 2, 0., z.data, 2);
//>                                                                   z.dump();
//>cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 2, 5, 1., x.data, 5, y.data, 2, 0., z.data, 2);
//>                                                                   z.dump();
//
//ut                                                    llas/gemm noTrans Trans
//>         Tensor<double> x(3,5); x.rand_(); Tensor<double> y(2,5); y.rand_();
//>                                                      Tensor<double> z(3,2);
//>  llas::gemm(false, true, 3, 2, 5, 1., x.data, 5, y.data, 5, 0., z.data, 2);
//>                                                                   z.dump();
//>cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 2, 5, 1., x.data, 5, y.data, 5, 0., z.data, 2);
//>                                                                   z.dump();
//
//ut                                                    llas/gemm Trans noTrans
//>         Tensor<double> x(5,3); x.rand_(); Tensor<double> y(5,2); y.rand_();
//>                                                      Tensor<double> z(3,2);
//>  llas::gemm(true, false, 3, 2, 5, 1., x.data, 3, y.data, 2, 0., z.data, 2);
//>                                                                   z.dump();
//>cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 2, 5, 1., x.data, 3, y.data, 2, 0., z.data, 2);
//>                                                                   z.dump();
//
//ut                                                      llas/gemm Trans Trans
//>         Tensor<double> x(5,3); x.rand_(); Tensor<double> y(2,5); y.rand_();
//>                                                      Tensor<double> z(3,2);
//>   llas::gemm(true, true, 3, 2, 5, 1., x.data, 3, y.data, 5, 0., z.data, 2);
//>                                                                   z.dump();
//>cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3, 2, 5, 1., x.data, 3, y.data, 5, 0., z.data, 2);
//>                                                                   z.dump();
}
// end   [[ lLAS ]] -----------------------------------------------------------



template <typename Dtype>
class Tensor {
public:
	// Optional name of the tensor
	std::string name;
	// Tensor shape, shape.size() = tensor dimension
	std::vector<size_t> shape;
	// Dynamic linear memory block for holding data
	Dtype* data = nullptr;

	// Destructor: common
	~Tensor(void) {
		shape.clear();
		if (nullptr != data) free(data);
	}
//ut                                                                   destruct
//>                                                             Tensor<bool> x;

	// setName: common
	void setName(string name) {
		this->name = name;
	}
//ut                                                                    setname
//>                                           Tensor<double> x; x.setName("x");

	// Constructor: empty
	Tensor(void) { }
//ut                                                                     create
//>                                              auto x = new Tensor<double>();
//>                                                                   delete x;

	// Constructor: 1D (e.g. vector) (from raw data)
	Tensor(size_t length, Dtype* mem = nullptr) {
		this->shape.push_back(length);
		this->resetMem();
		if (nullptr != mem) memcpy(data, mem, sizeof(Dtype)*length);
	}
//ut                                                         1d tensor creation
//>                                  Tensor<double> vector (10); vector.dump();

	// Constructor: 2D (e.g. matrix) (from raw data)
	Tensor(size_t row, size_t col, Dtype* mem = nullptr) {
		for (size_t i : {row, col}) this->shape.push_back(i);
		this->resetMem();
		if (nullptr != mem) memcpy(data, mem, sizeof(Dtype)*row*col);
	}
//ut                                                         2d tensor creation
//>                              Tensor<double> matrix (10, 10); matrix.dump();
//>                           auto x = new Tensor<double> (100, 100); delete x;

	// Constructor: 3D (e.g. image) (from raw data)
	Tensor(size_t channel, size_t height, size_t width, Dtype* mem = nullptr) {
		for (size_t i : {channel, height, width}) this->shape.push_back(i);
		this->resetMem();
		if (nullptr != mem) memcpy(data, mem, sizeof(Dtype)*channel*height*width);
	}
//ut                                                           create 3d tensor
//>                                                   Tensor<double> x (2,3,3);

	// Constructor: 4D (e.g. video) (from raw data)
	Tensor(size_t time, size_t channel, size_t height, size_t width, Dtype* mem = nullptr) {
		for (size_t i : {time, channel, height, width}) this->shape.push_back(i);
		this->resetMem();
		if (nullptr != mem) memcpy(data, mem, sizeof(Dtype)*time*channel*height*width);
	}
//ut                                                           create 4d tensor
//>                                                 Tensor<double> x (2,3,4,5);

	// Locator: 1D offset
	inline Dtype* at(size_t offset) {
		return this->data + offset;
	}
//ut                                                                  1d offset
//>                                 Tensor<int> x (10); ++*x.at(0); ++*x.at(9);

	// Locator: 2D offset
	inline Dtype* at(size_t row, size_t col) {
		return this->data + row*shape[1] + col;
	}
//ut                                                                  2d offset
//>                                                       Tensor<int> x(10,10);
//>                     ++*x.at(0,0); ++*x.at(0,9); ++*x.at(9,0); ++*x.at(9,9);

	// Locator: 3D offset
	inline Dtype* at(size_t c, size_t h, size_t w) {
		return this->data + c*shape[1]*shape[2] + h*shape[2] + w;
	}
//ut                                                                  3d offset
//>                                                    Tensor<int> x(10,10,10);
//>                                             ++*x.at(0,0,0); ++*x.at(9,9,9);

	// locator: 4D offset
	inline Dtype* at(size_t t, size_t c, size_t h, size_t w) {
		return this->data + t*shape[1]*shape[2]*shape[3] +
			c*shape[2]*shape[3] + h*shape[3] + w;
	}

	// Copier: common, copy raw data into this Tensor
	void copy(Dtype* mem, size_t sz) {
		if (sz > getSize())
			fprintf(stderr, "Tensor::copy Error: getSize()=%ld but you want to put data of size %ld.\n", getSize(), sz);
		assert(sz <= getSize());
		memcpy(data, mem, sizeof(Dtype)*sz);
	}

	// Copier: common, copy data from another instance
	void copy(Tensor<Dtype>* x) {
		assert(getSize() == x->getSize());
		memcpy(data, x->data, getSize());
	}

	// Slicer: 1D, slice of linear data [start,end), (non-inplace)
	// XXX: mem leak if forgot to delete
	Tensor<Dtype>* slice(size_t start, size_t end) {
		assert(end <= getSize()); assert(start <= end);
		return new Tensor<Dtype>(end-start, data+start);
	}

	// Dumper: Dump all the data shipped by this tensor to the screen
	// in a pretty format. The printing format is supported by the Julia
	// interpreter so you can copy the data to Julia.
	static inline void _edump(int    x) { printf(" \x1b[36m% d\x1b[m",  x); }
	static inline void _edump(long   x) { printf(" \x1b[36m% ld\x1b[m", x); }
	static inline void _edump(float  x) { printf(" \x1b[36m% .2f\x1b[m",  x); }
	static inline void _edump(double x) { printf(" \x1b[36m% .3lf\x1b[m", x); }
	void dump() {
		if (shape.size() == 0) {
			std::cout << "[ ]" << std::endl << "Tensor(,)" << std::endl;
			std::cout << "Tensor of name \"\x1b[36m" << name << "\x1b[m\", shape \x1b[36m(,)\x1b[m"
			   << std::endl;
		} else if (shape.size() == 1) {
			std::cout << "[";
			for (size_t i = 0; i < this->getSize(0); i++)
				_edump(*this->at(i));
			std::cout << " ]" << std::endl;
			std::cout << "Tensor of name \"\x1b[36m" << name << "\x1b[m\", shape \x1b[36m("
			   << this->getSize(0) <<  ",)\x1b[m" << std::endl;
		} else if (shape.size() == 2) {
			std::cout << "[" << std::endl;;
			for (size_t i = 0; i < this->getSize(0); i++) {
				std::cout << "  [";
				for (size_t j = 0; j < this->getSize(1); j++) {
					_edump(*this->at(i,j));
				}
				std::cout << " ]" << std::endl;
			}
			std::cout << "]" << std::endl;
			std::cout << "Tensor of name \"\x1b[36m" << name << "\x1b[m\", shape \x1b[36m("
			   << this->getSize(0) <<  "," << this->getSize(1) << ")\x1b[m"
			   << std::endl;
		} else if (shape.size() == 3) {
			std::cout << "[" << std::endl;
			for (size_t chan = 0; chan < shape[0]; chan++) {
				std::cout << "  [" << std::endl;
				for (size_t h = 0; h < shape[1]; h++) {
					std::cout << "    [";
					for (size_t w = 0; w < shape[2]; w++) {
						_edump(*this->at(chan,h,w));
					}
					std::cout << " ]" << std::endl;
				}
				std::cout << "  ]," << std::endl;
			}
			std::cout << "]" << std::endl;
			std::cout << "Tensor of name \"\x1b[36m" << name << "\x1b[m\", shape \x1b[36m("
			   << this->getSize(0) <<  "," << this->getSize(1) << ","
			   << getSize(2) << ")\x1b[m" << std::endl;
		} else if (shape.size() == 4) {
			std::cout << "[" << std::endl;
			for (size_t t = 0; t < shape[0]; t++) {
			std::cout << "  [" << std::endl;
			for (size_t c = 0; c < shape[1]; c++) {
			std::cout << "    [" << std::endl;
			for (size_t h = 0; h < shape[2]; h++) {
			std::cout << "      [";
			for (size_t w = 0; w < shape[3]; w++) {
				_edump(*this->at(t,c,h,w));
			}
			std::cout << " ]" << std::endl;
			}
			std::cout << "    ]," << std::endl;
			}
			std::cout << "  ]," << std::endl;
			}
			std::cout << "]" << std::endl;
			std::cout << "Tensor of name \"\x1b[36m" << name << "\x1b[m\", shape \x1b[36m("
			   << this->getSize(0) <<  "," << this->getSize(1) << ","
			   << getSize(2) << "," << getSize(3) << ")\x1b[m" << std::endl;
		} else {
			fprintf(stderr, "Tensor<*>::dump() for %ld-D tensor not implemented.\n", getDim());
		}
	}
//ut                                                             4d tensor dump
//>                            Tensor<double> X (2,3,5,5); X.rand_(); X.dump();
	//
//ut                                 float tensor, long tensor, int tensor dump
//>                                        Tensor<int> Xint (2,2); Xint.dump();
//>                                     Tensor<long> Xlong (2,2); Xlong.dump();
//>                                      Tensor<float> Xf32 (2,2); Xf32.dump();
//>                                     Tensor<double> Xf64 (2,2); Xf64.dump();

	// getDimension: common
	size_t getDim(void) const {
		return shape.size();
	}

	// getSize: common, size of the linear memory
	size_t getSize(void) const {
		if (shape.empty()) return 0;
		size_t size = 1;
		for (auto i: shape) size *= i;
		return size;
	}

	// getSize: common, the i-th dimension
	size_t getSize(size_t i) const {
		return (i >= shape.size()) ? -1 : shape[i];
	}

	// resetMem: common, reset the linear memory space
	void resetMem() {
		if (data != nullptr) free(data);
		data = nullptr;
		data = (Dtype*)malloc(sizeof(Dtype)*getSize());
		memset(data, 0x0, sizeof(Dtype)*getSize());
	}

	// Resizer: resize this instance from *D to 1D
	Tensor<Dtype>* resize(size_t length) {
		shape.clear();
		shape.push_back(length);
		resetMem();
		return this;
	}
//ut                                                   resize from empty tensor
//>                                                       Tensor<double> empty;
//>                                                           empty.resize(10);
//>                                                               empty.dump();
//>                                                       empty.resize(10, 10);
//>                                                               empty.dump();
//>                                                            empty.resize(1);
//>                                                               empty.dump();
//>                                                            empty.resize(0);

	// Resizer: resize this instance from *D to 2D
	Tensor<Dtype>* resize(size_t row, size_t col) {
		shape.clear();
		for (auto i : {row, col}) shape.push_back(i);
		resetMem();
		return this;
	}

	// Resizer: resize this instance from *D to 3D
	Tensor<Dtype>* resize(size_t c, size_t h, size_t w) {
		shape.clear();
		for (auto i : {c, h, w}) shape.push_back(i);
		resetMem();
		return this;
	}

	// Resizer: resize this instance from *D to 4D
	Tensor<Dtype>* resize(size_t t, size_t c, size_t h, size_t w) {
		shape.clear();
		for (auto i : {t, c, h, w}) shape.push_back(i);
		resetMem();
		return this;
	}

	// Resizer: resize this instance from *D to *D
	Tensor<Dtype>* resize(std::vector<size_t>& sz) {
		shape.clear();
		for (auto i : sz) shape.push_back(i);
		resetMem();
		return this;
	}

	// Resizer: resize this instance as another instance
	Tensor<Dtype>* resizeAs(Tensor<Dtype>* x) {
		this->resize(x->shape);
		return this;
	}

	// Resizer: expand: repeat a vector for many times
	// (d,)->(d,n), (d,1)->(d,n)
	// optional transpose, (d,)->(n,d), (d,1)->(n,d)
	// NOTE: don't forget to delete
	Tensor<Dtype>* expand(size_t N, bool trans = false) {
		size_t D = this->getSize();
		auto y = new Tensor<Dtype> ();
		if (!trans) { // no trans
			y->resize(D, N);
			for (size_t i = 0; i < D; i++)
				for (size_t j = 0; j < N; j++)
					*y->at(i,j) = data[i];
		} else { // trans == true
			y->resize(N, D);
			for (size_t i = 0; i < N; i++)
				for (size_t j = 0; j < D; j++)
					*y->at(i,j) = data[j];
		}
		return y;
	}
//ut                                                  expand w or w/o transpose
//>                                  Tensor<double> x (5); x.rand_(); x.dump();
//>                                  auto y = x.expand(3); y->dump(); delete y;
//>                            auto z = x.expand(3, true); z->dump(); delete z;

	// Resizer: unexpand: fold a matrix into a vector
	// axis = 1 -> (d,n) -> (d,)
	// axis = 0 -> (n,d) -> (d,)
	Tensor<Dtype>* unexpand(size_t axis = 1) {
		size_t D = (axis==1) ? shape[0] : shape[1];
		size_t N = (axis==1) ? shape[1] : shape[0];
		auto y = new Tensor<Dtype> (D); y->zero_();
		if (axis == 1) {
			for (size_t i = 0; i < D; i++)
				for (size_t j = 0; j < N; j++)
					*y->at(i) += *at(i,j);
		} else if (axis == 0) {
			for (size_t i = 0; i < N; i++)
				for (size_t j = 0; j < D; j++)
					*y->at(j) += *at(i,j);
		} else {
			assert(false);
		}
		return y;
	}
//ut                                                 unexpand by different axis
//>                                Tensor<double> x (3,5); x.rand_(); x.dump();
//>                                auto y = x.unexpand(1); y->dump(); delete y;
//>                                auto z = x.unexpand(0); z->dump(); delete z;

	// Filler: inplace, fill with zero for *D tensor
	Tensor<Dtype>* zero_() {
		memset(data, 0x0, sizeof(Dtype)*getSize());
		return this;
	}

	// Filler: inplace, fill with constant for *D tensor
	Tensor<Dtype>* fill_(Dtype value) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data,value)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = (Dtype) value;
		return this;
	}
//ut                                                               inplace fill
//>                                                        Tensor<double> ones;
//>                                            ones.resize(10, 10)->fill_(4.2);
//>                                                                ones.dump();

	// BLAS: inplace, scaling by a factor for *D tensor
	Tensor<Dtype>* scal_(Dtype factor) {
		llas::scal(getSize(), factor, data, 1);
		return this;
	}
//ut                                                               inplace scal
//>                                                        Tensor<double> ones;
//>                                            ones.resize(10, 10)->fill_(4.2);
//>                                                            ones.scal_(0.5);
//>                                                                ones.dump();

	// BLAS: inplace, *D, add constant to tensor
	void add_(Dtype constant) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data,constant)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) += constant;
	}

	// BLAS: inplace, *D, add *D tensor to *D tensor
	void add_(Tensor<Dtype>* X) {
		assert(getSize() == X->getSize());
		llas::axpy(getSize(), (Dtype)1., X->data, 1, data, 1);
	}

	// BLAS: SUM, *D, sum_i x_i
	Dtype sum(void) {
		Dtype ret = 0.;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret)
#endif
		for (size_t i = 0; i < getSize(); i++)
			ret += *at(i);
		return ret;
	}

	// BLAS: ASUM, *D, sum_i |x_i|
	Dtype asum(void) {
		return llas::asum(getSize(), data, 1);
	}

	// BLAS, inplace, *D, y_i = exp(x_i)
	Tensor<Dtype>* exp_(void) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = std::exp(*(data + i));
		return this;
	}

	// MISC: inplace Sqrt, *D, x_i <- sqrt(x_i)
	Tensor<Dtype>* sqrt_(void) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = std::sqrt(*(data + i));
		return this;
	}
//ut                                                               inplace sqrt
//>                            Tensor<double> x (10, 10); x.rand_(); x.sqrt_();

	// MISC: inplace, *D, rand ~U[0.0, 1.0)
	Tensor<Dtype>* rand_(void) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = drand48();
		return this;
	}
//ut                                                             inplace random
//>                                                   Tensor<double> x (5, 10);
//>                                                                  x.rand_();
//>                                                                   x.dump();

	// MISC: inplace, *D, rand ~U[l, u)
	Tensor<Dtype>* uniform_(Dtype l, Dtype u) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data, l, u)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = drand48() * (u-l) + l;
		return this;
	}
//ut                                                                   uniform_
//>                                                  Tensor<double> x (10, 10);
//>                                                        x.uniform_(-10, 10);
//>                                                                   x.dump();

	// MISC: MAE: y <- sum_i ||a_i - b_i||_1
	Dtype MAE(Tensor<Dtype>* B) {
		assert(this->getSize() == B->getSize());
		Dtype ret = 0.;
		size_t size = this->getSize();
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret) shared(size)
#endif
		for (size_t i = 0; i < size; i++) {
			Dtype tmp = *this->at(i) - *B->at(i);
			ret += (tmp > (Dtype)0.) ? tmp : -tmp;
		}
		return ret / (Dtype)this->getSize();
	}

	// MISC: MSE: y <- sum_i ||a_i - b_i||_2^2
	Dtype MSE(Tensor<Dtype>* B) {
		assert(this->getSize() == B->getSize());
		Dtype ret = 0.;
		size_t size = this->getSize();
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret) shared(size)
#endif
		for (size_t i = 0; i < size; i++) {
			Dtype tmp = *this->at(i) - *B->at(i);
			ret += tmp * tmp;
		}
		return ret / (Dtype)this->getSize();
	}

	// MISC: rot180, non-inplace
	// XXX: don't forget to delete!
	Tensor<Dtype>* rot180(void) {
		Tensor<Dtype>* rot = this->clone();
		auto eswap = [](Dtype* a, Dtype* b) {
			Dtype tmp = *a; *a = *b; *b = tmp;
		};
		if (getDim() == 1 || getDim() == 2) {
			size_t curl = 0, curr = rot->getSize()-1;
			while (curl < curr) eswap(rot->at(curl++), rot->at(curr--));
		} else if (getDim() == 3) {
			for (size_t c = 0; c < shape[0]; c++) {
				size_t HxW = rot->shape[1] * rot->shape[2];
				size_t curl = c * HxW, curr = (c+1) * HxW - 1;
				while (curl < curr) eswap(rot->at(curl++), rot->at(curr--));
			}
		} else if (getDim() == 4) {
			for (size_t t = 0; t < shape[0]; t++) {
				size_t CxHxW = rot->shape[1] * rot->shape[2] * rot->shape[3];
				for (size_t c = 0; c < shape[1]; c++) {
					size_t HxW = rot->shape[2] * rot->shape[3];
					size_t curl = t*CxHxW + c*HxW;
					size_t curr = t*CxHxW + (c+1)*HxW - 1;
					while (curl < curr) eswap(rot->at(curl++), rot->at(curr--));
				}
			}
		} else {
			fprintf(stderr, "Tensor::rot180 not implemented for %ld-D tensor!\n", getDim());
			exit(EXIT_FAILURE);
		}
		return rot;
	}
//ut                                                                     rot180
//>                                 Tensor<double> X (10); X.rand_(); X.dump();
//>                                auto xx = X.rot180(); xx->dump(); delete xx;
//>                                        X.resize(5, 5); X.rand_(); X.dump();
//>                                     xx = X.rot180(); xx->dump(); delete xx;
//>                                     X.resize(3, 5, 5); X.rand_(); X.dump();
//>                                     xx = X.rot180(); xx->dump(); delete xx;
//>                                     X.resize(2,2,5,5); X.rand_(); X.dump();
//>                                     xx = X.rot180(); xx->dump(); delete xx;

	// Clone: deep copy of this Tensor
	// XXX: mem leaks if forgot to delete!
	Tensor<Dtype>* clone(void) {
		Tensor<Dtype>* y = new Tensor<Dtype> ();
		y->resizeAs(this);
		//#pragma omp parallel for // slower than memcpy
		//for (size_t i = 0; i < getSize(); i++) *(y->data + i) = *(data + i);
		memcpy(y->data, this->data, sizeof(Dtype)*this->getSize()); // BEST
		return y;
	}
//ut                                                                      clone
//>                                                  Tensor<double> x (10, 10);
//>                                                                  x.rand_();
//>                                                                   x.dump();
//>                                              Tensor<double>* y = x.clone();
//>                                                                  y->dump();
//>                                             cout << &x << " " << y << endl;
//>                                                                   delete y;

	// MISC: sign: apply the sign function to the tensor
	// XXX: remember to delete the created tensor.
	Tensor<Dtype>* sign(void) {
		Tensor<Dtype>* y = this->clone();
#if defined(USE_OPENMP)
#pragma omp parallel for
#endif
		for (size_t i = 0; i < getSize(); i++)
			y->data[i] = (y->data[i] > 0.) ? 1.
				: (y->data[i] < 0.) ? -1. : 0.;
		return y;
	}

	// Transpose: 2D transpose, non-inplace
	// XXX: don't forget to delete
	Tensor<Dtype>* transpose(void) {
		if (shape.size() != 2) {
			fprintf(stderr, "transpose(): ERROR: shape.size = %ld\n", shape.size());
			exit(EXIT_FAILURE);
		}
		auto xT = new Tensor<Dtype> ((size_t)shape[1], (size_t)shape[0]);
		#if defined(USE_OPENMP)
		#pragma omp parallel for collapse(2)
		#endif
		for (size_t i = 0; i < shape[0]; i++)
			for (size_t j = 0; j < shape[1]; j++)
				*xT->at(j, i) = *at(i, j);
		return xT;
	}

	// Transpose: 2D transpose in-place
	void transpose_(bool ushape=true) {
		assert(getDim() == 2);
		size_t oldcol = shape[1];
		size_t newrow = shape[1], newcol = shape[0];
		vector<bool> visited (getSize(), false);
		Dtype c = 0., t = 0.;
		for (size_t n = 0; n < getSize(); n++) {
			if (visited[n]) continue;
			visited[n] = true;
			size_t srcrow = n / oldcol, srccol = n % oldcol;
			size_t dstrow = srccol, dstcol = srcrow;
			size_t next = dstrow * newcol + dstcol;
			c = *(data + n);
			while (!visited[next]) {
				visited[next] = true;
				t = *(data + next); *(data + next) = c; c = t;
				srcrow = next / oldcol, srccol = next % oldcol;
				dstrow = srccol, dstcol = srcrow;
				next = dstrow * newcol + dstcol;
			}
			*(data + n) = c;
		}
		if (ushape) { shape[0] = newrow; shape[1] = newcol; }
	}
//ut                                                   tensor transpose inplace
//>                                Tensor<double> X (5,3); X.rand_(); X.dump();
//>                                                   X.transpose_(); X.dump();

	// DEBUG: compares size of two tensors
	bool sameSize(Tensor<Dtype>* x) {
		if (x->getDim() != getDim()) return false;
		for (size_t i = 0; i < shape.size(); i++)
			if (x->getSize(i) != getSize(i)) return false;
		return true;
	}
//ut                                                               int sameSize
//>                                                     Tensor<int> x (10, 10);
//>                                                          Tensor<int> y (1);
//>                                                     Tensor<int> z (10, 11);
//>                                             assert(x.sameSize(&x) == true);
//>                                            assert(x.sameSize(&y) == false);
//>                                            assert(x.sameSize(&z) == false);

	// I/O: save *D tensor to ASCII-encoded json file
	void save(string fname) {
		std::ofstream f (fname); assert(f);
		Json::Value jroot;
		Json::StyledWriter jwriter;
		// fill in the json object
		jroot["type"] = "Leicht::Tensor";
		jroot["version"] = LEICHT_VERSION;
		jroot["name"] = this->name;
		for (auto i : this->shape) jroot["shape"].append((const Json::Value::UInt64)i);
		for (size_t i = 0; i < getSize(); i++)
			jroot["data"].append((double)data[i]);
		// write
		f << jwriter.write(jroot) << endl;
		f.close();
	}
//ut                                                3D tensor creation and save
//>                                                 Tensor<double> x (3, 6, 6);
//>                                                                  x.rand_();
//>                                                                   x.dump();
//>                                                      x.save("test.leicht");

	// I/O: load *D tensor from ASCII-encoded json file
	void load(string fname) {
		Json::Value jroot;
		Json::Reader jreader;
		std::fstream f (fname, ios::in); assert(f);
		jreader.parse(f, jroot);
		assert(jroot["type"] == "Leicht::Tensor");
		// read data in
		this->name = jroot["name"].asString();
		std::vector<size_t> sz;
		for (size_t i = 0; i < jroot["shape"].size(); i++)
			sz.push_back(jroot["shape"][(int)i].asUInt64());
		this->resize(sz);
		for (size_t j = 0; j < jroot["data"].size(); j++)
			data[j] = jroot["data"][(int)j].asDouble();
		// cleanup
		f.close();
	}
//ut                                                             3D tensor load
//>                                                           Tensor<double> x;
//>                                                      x.load("test.leicht");
//>                                                                   x.dump();

	// Shortcut: operator += in-place addition, element-wise
	void operator+= (Tensor<Dtype>& x) { Tensor<Dtype>::axpy((Dtype)1.,&x,this); }
//ut                                                                operator +=
//>           Tensor<double> x (5,5); x.fill_(2.1); x.dump(); x += x; x.dump();

	// Shortcut: operator -= in-place subtraction, element-wise
	void operator-= (Tensor<Dtype>& x) { Tensor<Dtype>::axpy((Dtype)-1.,&x,this); }
//ut                                                                operator -=
//>           Tensor<double> x (5,5); x.fill_(2.1); x.dump(); x -= x; x.dump();
	
	// Shortcut: operator *= in-place multiplication, element-wise
	void operator*= (Tensor<Dtype>& x) {
		assert(x.getSize() == this->getSize());
#if defined(USE_OPENMP)
#pragma omp parallel for
#endif
		for (size_t i = 0; i < this->getSize(); i++) *at(i) *= *x.at(i);
	}
//ut                                                                operator *=
//>           Tensor<double> x (5,5); x.fill_(2.1); x.dump(); x *= x; x.dump();
	
	// Shortcut: operator /= in-place division, element-wise
	// XXX: Be care of the division-by-zero issue.
	void operator/= (Tensor<Dtype>& x) {
		assert(x.getSize() == this->getSize());
#if defined(USE_OPENMP)
#pragma omp parallel for
#endif
		for (size_t i = 0; i < this->getSize(); i++) *at(i) /= *x.at(i);
	}
//ut                                                                operator /=
//>           Tensor<double> x (5,5); x.fill_(2.1); x.dump(); x /= x; x.dump();
	
	// Shortcut: operator + create new tensor c, add a and b to it
	// XXX: don't forget to delete
	Tensor<Dtype>* operator+ (Tensor<Dtype>& a) {
		auto ret = this->clone(); *ret += a; return ret;
	}
//ut                                                                 operator +
//>                             Tensor<double> x (5,5); x.fill_(2.1); x.dump();
//>                             Tensor<double> y (5,5); y.fill_(4.2); y.dump();
//>                   auto z = x + y; z->setName("z=x+y"); z->dump(); delete z;
	
	// Shortcut: operator > create mask tensor where this_i > a
	// XXX: don't forget to delete
	Tensor<Dtype>* operator> (Dtype a) {
		auto ret = this->clone(); 
#if defined(USE_OPENMP)
#pragma omp parallel for
#endif
		for (size_t i = 0; i < this->getSize(); i++) *ret->at(i) = (Dtype)(*at(i) > a);
		return ret;
	}
//ut                                                          operator > scalar
//>                     Tensor<double> x (5,5); x.rand_()->add_(-.5); x.dump();
//>                                       auto y = x > 0.; y->dump(); delete y;

	// LEVEL1 BLAS: AXPY (Tensor)
	static void
	axpy(Dtype alpha, Tensor<Dtype>* X, Tensor<Dtype>* Y) {
		// regard tensor as a flattened, check size
		assert(X->getSize() == Y->getSize());
		size_t sz = X->getSize();
		llas::axpy(sz, alpha, X->data, 1, Y->data, 1);
	}
//ut                                                                       AXPY
//>                                                  Tensor<double> x (10, 10);
//>                                                               x.fill_(2.1);
//>                                           Tensor<double>::axpy(1., &x, &x);
//>                                                                   x.dump();

	// (Tensor) GEMM
	static void
	gemm(bool transA, bool transB,
			double alpha, Tensor<double>* A, Tensor<double>* B,
			double beta, Tensor<double>* C)
	{
		// FIXME: Size Check
		//fprintf(stderr, "GEMM: Illegal Shape! (%ld,%ld)x(%ld,%ld)->(%ld,%ld)",
		size_t M, N, K, lda, ldb, ldc;
		if (!transA && !transB) {
			M = A->getSize(0); N = B->getSize(1); K = A->getSize(1);
		} else if (!transA && transB) {
			M = A->getSize(0); N = B->getSize(0); K = A->getSize(1);
		} else if (transA && !transB) {
			M = A->getSize(1); N = B->getSize(1); K = A->getSize(0);
		} else {
			M = A->getSize(1); N = B->getSize(0); K = A->getSize(0);
		}
		lda = transA ? M : K;
		ldb = transB ? K : N;
		ldc = N;
		llas::gemm(transA, transB, M, N, K,
				alpha, A->data, lda, B->data, ldb, beta, C->data, ldc);
	}
//ut                                                           openblas gemm NN
//>     Tensor<double> x (2,5); Tensor<double> y (5,2); Tensor<double> z (2,2);
//>                     Tensor<double>::gemm(false, false, 1., &x, &y, 0., &z);
//ut                                                           openblas gemm NT
//>     Tensor<double> x (2,5); Tensor<double> y (2,5); Tensor<double> z (2,2);
//>                      Tensor<double>::gemm(false, true, 1., &x, &y, 0., &z);
//ut                                                           openblas gemm TN
//>     Tensor<double> x (5,2); Tensor<double> y (5,2); Tensor<double> z (2,2);
//>                      Tensor<double>::gemm(true, false, 1., &x, &y, 0., &z);
//ut                                                           openblas gemm TT
//>     Tensor<double> x (5,2); Tensor<double> y (2,5); Tensor<double> z (2,2);
//>                       Tensor<double>::gemm(true, true, 1., &x, &y, 0., &z);


}; // end class Tensor

// DEBUG: shortcut to dump a Tensor
template <typename Dtype>
std::ostream& operator<< (std::ostream& out, Tensor<Dtype>& x) {
	x.dump();
	return out;
}


// MISC: Conv2D in 3 different modes: {valid,full,same}
// valid: X(H,W) * K(R,R) (accumulate)-> Y(H-R+1,W-R+1)
// full : X(H,W) * K(R,R) (accumulate)-> Y(H+R-1,W+R-1)
// same : FIXME
// Note, this "convolution" is for Computer Vision instead of math.
//
// <internal> the "valid" branch of Conv2D
template <typename Dtype>
static void
_Conv2D_valid(Dtype* X, size_t H, size_t W,
		Dtype* K, size_t R, Dtype* Y)
{
	size_t ik = 0, jk = 0; Dtype Yij = 0.;
// I5-2520M: 224x224, k=3, 30pass: 14ms (no OMP) 7ms (OMP)
// I7-6900K: 224x224, k=3, 30pass: 12ms (no OMP) 2ms (OMP)
// XXX: Valgrind will keep track of the threads so the parallel mode is
//      slower when running with valgrind.
#if defined(USE_OPENMP)
#pragma omp parallel for collapse(2) shared(X,H,W,K,R,Y) private(ik,jk,Yij)
#endif
	for (size_t i = 0; i < H-R+1; i++) { // i - row of dest feature map
		for (size_t j = 0; j < W-R+1; j++) { // j - column of dest feature map
			Yij = 0.;
			for (ik = 0; ik < R; ik++) { // ik - row of kernel
				for (jk = 0; jk < R; jk++) { // jk - column of kernel
					Yij += *(X + (i+ik)*W + (j+jk)) * *(K + ik*R + jk);
				}
			}
			*(Y + i*(W-R+1) + j) += Yij;
		}
	}
}
// <internal> the "full" branch of Conv2D
template <typename Dtype>
static void
_Conv2D_full(Dtype* X, size_t H, size_t W,
		Dtype* K, size_t R, Dtype* Y)
{
	size_t ik = 0, jk = 0; Dtype Yij = 0.;
#if defined(USE_OPENMP)
#pragma omp parallel for collapse(2) shared(X,H,W,K,R,Y) private(ik,jk,Yij)
#endif
	for (size_t i = 0; i < H+R-1; i++) { // i - row of dest feature map
		for (size_t j = 0; j < W+R-1; j++) { // j - column of dest feature map
			Yij = 0.;
			for (ik = 0; ik < R; ik++) { // ik - row of kernel
				for (jk = 0; jk < R; jk++) { // jk - column of kernel
// Yij += K(ik,jk) * X(-KH+1+i+ik, -KW+1+j+jk) if index valid
					Yij +=
((long)-R+1+i+ik>=0 && (long)-R+1+i+ik<H &&
 (long)-R+1+j+jk>=0 && (long)-R+1+j+jk<W) ?
*(X + ((long)-R+1+i+ik)*W + ((long)-R+1+j+jk)) * *(K + ik*R + jk) : (Dtype)0.;
				}
			}
			*(Y + i*(W+R-1) + j) += Yij;
		}
	}
}
// The Conv2D wrapper you should use
template <typename Dtype>
void
Conv2D(string mode,
		Dtype* X, size_t H, size_t W,
		Dtype* K, size_t R,
		Dtype* Y)
{
	if (mode == "valid") {
		_Conv2D_valid(X, H, W, K, R, Y);
	} else if (mode == "full") {
		_Conv2D_full(X, H, W, K, R, Y);
	} else {
		fprintf(stderr, "Conv2D: Invalid convolution mode!\n");
		exit(EXIT_FAILURE);
	}
}
//ut                                              convolution vallid mode tests
//>                          Tensor<double> X (3,3); X.setName("X"); X.rand_();
//>                        Tensor<double> K (2,2); K.setName("K"); K.fill_(1.);
//>                          Tensor<double> Y (2,2); Y.setName("Y"); Y.zero_();
//>                           Conv2D("valid", X.data, 3, 3, K.data, 2, Y.data);
//>                                                cout << X << K << Y << endl;
//
//>                                                   X.resize(5,5); X.rand_();
//>                                                 K.resize(3,3); K.fill_(1.);
//>                                                   Y.resize(3,3); Y.zero_();
//>                           Conv2D("valid", X.data, 5, 5, K.data, 3, Y.data);
//>                                                cout << X << K << Y << endl;
//
//>                                                 K.resize(2,2); K.fill_(1.);
//>                                                   Y.resize(4,4); Y.zero_();
//>                           Conv2D("valid", X.data, 5, 5, K.data, 2, Y.data);
//>                                                cout << X << K << Y << endl;
//
//>                                                 K.resize(1,1); K.fill_(1.);
//>                                                   Y.resize(5,5); Y.zero_();
//>                           Conv2D("valid", X.data, 5, 5, K.data, 1, Y.data);
//>                                                cout << X << K << Y << endl;
//
//>                                                 //X.resize(2,2); X.rand_();
//>                                               //K.resize(3,3); K.fill_(1.);
//>                                                 //Y.resize(2,2); Y.zero_();
//>                         //Conv2D("valid", X.data, 2, 2, K.data, 3, Y.data);
//>                                              //cout << X << K << Y << endl;
//>     //XXX: doesn't work for BIG kernel. same behaviour as in Octave (conv2)
//
//ut                                                 convolution full mode test
//>                          Tensor<double> X (3,3); X.setName("X"); X.rand_();
//>                        Tensor<double> K (2,2); K.setName("K"); K.fill_(1.);
//>                          Tensor<double> Y (4,4); Y.setName("Y"); Y.zero_();
//>                            Conv2D("full", X.data, 3, 3, K.data, 2, Y.data);
//>                                                cout << X << K << Y << endl;
//>                                                                            
//>                                                   X.resize(5,5); X.rand_();
//>                                                 K.resize(3,3); K.fill_(1.);
//>                                                   Y.resize(7,7); Y.zero_();
//>                            Conv2D("full", X.data, 5, 5, K.data, 3, Y.data);
//>                                                cout << X << K << Y << endl;
//>                                                                            
//>                                                 K.resize(1,1); K.fill_(1.);
//>                                                   Y.resize(5,5); Y.zero_();
//>                            Conv2D("full", X.data, 5, 5, K.data, 1, Y.data);
//>                                                cout << X << K << Y << endl;

//ut                                                   feed hdf5 data to tensor
//>                                            Tensor<double> matrix (10, 784);
//>                                                 Tensor<double> vector (10);
//>                                                      vector.name = "label";
//>            leicht_hdf5_read("demo.h5", "data", 0, 0, 10, 784, matrix.data);
//>                   leicht_hdf5_read("demo.h5", "label", 0, 10, vector.data);
//>                                                              vector.dump();

#endif // defined(_LEICHT_TENSOR_HPP)
