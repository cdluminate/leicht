/* benchmarks for Leicht :: tensor.hpp
 */
#include "leicht.hpp"
#include <cblas-openblas.h>
using namespace std;

/*
 * 10 * double gemm (512,512)
 *   5400 ms, tensor
 *   90   ms, openblas
 *   60x
 *
 * 10 * float gemm (512,512)
 *   1100 ms, tensor
 *   50   ms, openblas
 *   22x
 */

int
main(void)
{

	cout << "Leicht::Tensor::GEMM<double>, 10 times" << endl;
	{
		Tensor<double> x (512, 512); x.rand_();
		Tensor<double> y (512, 512); y.rand_();
		Tensor<double> z (512, 512); z.zero_();

		tic();
		for (int i = 0; i < 100; i++) {
			Tensor<double>::gemm(false, false, 1., &x, &y, 0., &z);
		}
		toc();
		tic();
		for (int i = 0; i < 100; i++) {
			Tensor<double>::gemm(false,  true, 1., &x, &y, 0., &z);
		}
		toc();
		tic();
		for (int i = 0; i < 100; i++) {
			Tensor<double>::gemm( true, false, 1., &x, &y, 0., &z);
		}
		toc();
		tic();
		for (int i = 0; i < 100; i++) {
			Tensor<double>::gemm( true,  true, 1., &x, &y, 0., &z);
		}
		toc();
	}

	cout << "OpenBLAS::GEMM<double>, 10 times" << endl;
	{
		Tensor<double> x (512, 512); x.rand_();
		Tensor<double> y (512, 512); y.rand_();
		Tensor<double> z (512, 512); z.zero_();

		tic();
		for (int i = 0; i < 100; i++) {
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				512, 512, 512, 1., x.data, 512, y.data, 512, 0., z.data, 512);
		}
		toc();
		tic();
		for (int i = 0; i < 100; i++) {
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				512, 512, 512, 1., x.data, 512, y.data, 512, 0., z.data, 512);
		}
		toc();
		tic();
		for (int i = 0; i < 100; i++) {
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				512, 512, 512, 1., x.data, 512, y.data, 512, 0., z.data, 512);
		}
		toc();
		tic();
		for (int i = 0; i < 100; i++) {
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
				512, 512, 512, 1., x.data, 512, y.data, 512, 0., z.data, 512);
		}
		toc();
	}

//	cout << "Leicht::Tensor::GEMM<float>, 10 times" << endl;
//	{
//		Tensor<float> x (512, 512); x.rand_();
//		Tensor<float> y (512, 512); y.rand_();
//		Tensor<float> z (512, 512); z.zero_();
//		tic();
//		for (int i = 0; i < 100; i++) {
//			Tensor<double>::gemm(false, false, 1., &x, &y, 0., &z);
//		}
//		toc();
//	}

//	cout << "OpenBLAS::GEMM<float>, 10 times" << endl;
//	{
//		Tensor<float> x (512, 512); x.rand_();
//		Tensor<float> y (512, 512); y.rand_();
//		Tensor<float> z (512, 512); z.zero_();
//		tic();
//		for (int i = 0; i < 100; i++) {
//			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//				512, 512, 512, 1., x.data, 512, y.data, 512, 0., z.data, 512);
//		}
//		toc();
//	}

	return 0;
}

// [[[ Benchmarks ]]]

//ut                                       GEMM Benchmark double 512x512, 2pass
//>                                      Tensor<double> X (512,512); X.rand_();
//>                                      Tensor<double> K (512,512); K.rand_();
//>                                                 Tensor<double> Y (512,512);
//>             tic(); for (int i = 0; i < 2; i++) GEMM(1.,&X,&K,0.,&Y); toc();
//
//ut                                               valid  convolution benchmark
//>                                     Tensor<double> X (224, 224); X.rand_();
//>                                          Tensor<double> K (3,3); K.rand_();
//>                                                Tensor<double> Y (222, 222);
//>                                                                      tic();
//>                                              for (int i = 0; i < 3*10; i++)
//>                       Conv2D("valid", X.data, 224, 224, K.data, 3, Y.data);
//>                                                                      toc();

//ut                                                        transpose benchmark
//>                                     Tensor<double> X (512,1024); X.rand_();
//>  tic(); for (int i=0;i<10;i++) { auto y = X.transpose(); delete y; } toc();

//ut                                                            clone benchmark
//>                                       Tensor<double> X(512,512); X.rand_();
//>     tic(); for(int i=0;i<1000;i++) { auto y = X.clone(); delete y; } toc();
//
//ut                              openblas GEMM Benchmark double 512x512, 2pass
//>                                      Tensor<double> X (512,512); X.rand_();
//>                                      Tensor<double> K (512,512); K.rand_();
//>                                                 Tensor<double> Y (512,512);
//>tic(); for (int i = 0; i < 2; i++) GEMM(false, false, 1.,&X,&K,0.,&Y); toc();
//
