/* layer.cc for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#if !defined(_LEICHT_LAYER_HPP)
#define _LEICHT_LAYER_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tensor.hpp"
#include "blob.hpp"

using namespace std;

// Basic Layer Class
template <typename Dtype>
class Layer {
public:
	// Optional name of a layer
	string name;
	// list of learnable parameters
	std::vector<Blob<Dtype>*> parameters;

	// The constructor should only require the shape of the parameter,
	// because the output blob shape could be dynamically determined
	// using the input blob shape and parameter blob shape when forwarding.
	// Specifically:
	//   1. Setup the special attributes of the layer
	//   2. Setup the learnable parameters
	//   3. Fill the parameter blobs with initial value.
	//   4. Register the parameters.
	Layer(void) { }

	// zero the gradient of the parameters. Do nothing if the layer ships
	// no learnable parameter. Does not zero the bottom and top gradient.
	// Specifically:
	//   1. Apply ::zeroGrad on all the internal parameter blobs
	void zeroGrad(void) {
		for (auto iter = parameters.begin(); iter != parameters.end(); iter++) {
			Blob<Dtype>* param = *iter;
			param->gradient.zero_();
		}
	}

	// conduct forward pass with bottom and top blobs. We assume that the
	// user has already correctly setup the blob size. The output blob size
	// could be derived from the input shape and the parameter blob shape
	// dynamically when forwarding. This makes dynamic graph more possible.
	// Specifically:
	void forward() { }

	// Just copy the the input blob to the output blob, this identity
	// transformation can be used to e.g. reshape a blob.
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		output.value.copy(input.value.data, input.value.getSize());
	}

	// conduct backward pass with bottom and top blobs.
	// gradient should be accumulated into the bottom.gradient tensor,
	// and bottom.gradient should not be overriten by a backward function
	// if possible. Such that multi-branch and recurrent could be possible.
	// The backward pass should be aware of the requires_grad attribute.
	void backward() { }

	// conditionally copy the output.gradient to input.gradient since the
	// forward pass conducts identity transformation.
	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		if (input.requires_grad && output.requires_grad)
		input.gradient.copy(output.gradient.data, output.gradient.getSize());
	}

	// apply first-order optimization method on learnable parameters if any.
	void update(double lr, string optim="SGD") {
		if (optim == "SGD") this->SGD(lr);
		else if (optim == "SGDM") this->SGDM(lr);
		else if (optim == "Adam") this->Adam(lr);
		else {
			fprintf(stderr, "Unknown Optimizer!\n");
			assert(false);
		}
   	}

	// Optim: Stochastic Gradient Descent
	void SGD(double lr) {
		for (auto iter = parameters.begin(); iter != parameters.end(); iter++) {
			Blob<Dtype>* param = *iter;
			Tensor<Dtype>::axpy((Dtype)-lr, &param->gradient, &param->value);
		}
	}

	// Optim: Stochastic Gradient Descent with Momentum
	std::vector<Tensor<Dtype>*> _sgdm_v; // NOTE: delete tensors on destruction
	void SGDM(double lr, double momentum=0.9) {
		if (_sgdm_v.size() != parameters.size()) {
			for (size_t i = 0; i < parameters.size(); i++) {
				Tensor<Dtype>* v = parameters[i]->gradient.clone();
				v->zero_();
				_sgdm_v.push_back(v);
			}
		}
		// v <- av - lr * dW, W <- W + v
		for (size_t i = 0; i < parameters.size(); i++) {
			Blob<Dtype>* param = parameters[i];
			Tensor<Dtype>* v   = _sgdm_v[i];
			v->scal_(momentum);
			Tensor<Dtype>::axpy((Dtype)-lr, &param->gradient, v);
			Tensor<Dtype>::axpy((Dtype)1., v, &param->value);
		}
	}

	// FIXME: Adam still doesn't work?
	// Optim: Adam, SGD with Adaptive learning rate
	// @ref Deep Learning Book, Ian Goodfwllow, et al. p. 311
	size_t _adam_t = 0;
	std::vector<Tensor<Dtype>*> _adam_s; // NOTE: delete tensors on destruction
	std::vector<Tensor<Dtype>*> _adam_r; // NOTE: delete tensors on destruction
	void Adam(double lr, double rho1 = 0.9, double rho2 = 0.999,
			double epsilon = 1e-8)
	{
		if (_adam_s.size() != parameters.size()) {
			for (size_t i = 0; i < parameters.size(); i++) {
				Tensor<Dtype>* s = parameters[i]->gradient.clone();
				s->zero_();
				_adam_s.push_back(s);
				Tensor<Dtype>* r = parameters[i]->gradient.clone();
				r->zero_();
				_adam_r.push_back(r);
			}
		}
		// update t
		_adam_t++;
		// update parameter
		for (size_t i = 0; i < parameters.size(); i++) {
			Blob<Dtype>* param = parameters[i]; // parameter
			Tensor<Dtype>* s = _adam_s[i];
			Tensor<Dtype>* r = _adam_r[i];
			// update biased first moment estimate
			s->scal_(rho1);
			auto g = param->gradient.clone();
			g->scal_(1. - rho1);
			*s += *g;
			delete g;
			// update biased second moment estimate
			r->scal_(rho2);
			auto g2 = param->gradient.clone();
			*g2 *= *g2;
			g2->scal_(1. - rho2);
			*r += *g2;
			delete g2;
			// correct bias in first moment
			s->scal_(1. / (1. - pow(rho1, _adam_t)));
			// correct bias in second moment
			r->scal_(1. / (1. - pow(rho2, _adam_t)));
			// compute update
			r->sqrt_()->add_(epsilon);
			*s /= *r;
			s->scal_((Dtype)-lr);
			param->value += *s;
		}
	}

	// Apply regularization on the parameters and store the gradient
	// FIXME: return the regularization loss value?
	void regularization(string type = "L2", Dtype weight = 1e-5) {
		if (type == "L2") _L2_regularization(weight);
		else if (type == "L1") _L1_regularization(weight);
		else {
			fprintf(stderr, "Regularization: Unknown type!\n");
			exit(EXIT_FAILURE);
		}
	}

	// Loss[:Regularization]_{L_2} = \lambda ||W||^2_2
	void _L2_regularization(Dtype weight = 1e-5) {
		for (auto iter = parameters.begin(); iter != parameters.end(); iter++) {
			Blob<Dtype>* param = *iter;
			Tensor<Dtype>::axpy((Dtype)2. * weight, &param->value, &param->gradient);
		}
	}

	// Loss[:Regularization]_{L_1} = \lambda \sum |w_i|
	void _L1_regularization(Dtype weight = 1e-5) {
		for (auto iter = parameters.begin(); iter != parameters.end(); iter++) {
			Blob<Dtype>* param = *iter;
			Tensor<Dtype>* l1grad = param->value.sign();
			Tensor<Dtype>::axpy((Dtype)weight, l1grad, &param->gradient);
			delete l1grad;
		}
	}

};
//ut                                                           layer new delete
//>                        auto l = new LinearLayer<double> (10, 10); delete l;
//
//ut                                                             identity layer
//>                                                  Blob<double> X (100, 784);
//>                                               Blob<double> Y (100, 28, 28);
//>                                                          Layer<double> id1;
//>                                                          id1.forward(X, Y);
//>                                                         id1.backward(X, Y);


/* Layers we currently have:
 * 1. Linear Layer
 * 2. Conv2d Layer
 * 3. Relu Layer
 * 4. Softmax Layer
 * 5. MSE Loss Layer
 * 6. Classification negative log likelihood Loss layer (ClassNLLLoss)
 * 7. Classification accyracy Layer
 * 8. Maxpool Layer
 * 9. Transpose Layer
 */

template <typename Dtype>
class LinearLayer : public Layer<Dtype> {
public:
	Blob<Dtype> W; // weight matrix
	Blob<Dtype> b; // bias vector
	bool use_bias = true;
	bool row_major = true;

	// Cached shape info
	// 1. input: (N,D) or (D,N) or (N,...) or (...,N)
	// 2. weight: !row_major ? (K,D) or (K,...) : (D,K) or (...,K)
	// 3. bias:  (K,)
	// N batchsize, D input dim, K output dim
	
	LinearLayer(size_t dim_dest, size_t dim_src,
			bool use_bias=true, bool row_major=false) {
		// setup attrib
		this->use_bias = use_bias;
		this->row_major = row_major;
		// setup weight
		if (!row_major) W.resize(dim_dest, dim_src);
		else W.resize(dim_src, dim_dest);
		W.setName("LinearLayer/W");
		W.gradient.zero_();
		// setup bias
		if (use_bias) b.resize(dim_dest);
		if (use_bias) b.setName("LinearLayer/b");
		if (use_bias) b.gradient.zero_();
		// parameter initialization
		// @ref Torch:nn, W,b ~ uniform(-stdv, stdv)
		//     where stdv = 1. / sqrt(inputSize)
		double stdv = 1. / std::sqrt(dim_src);
		W.value.uniform_(-stdv, stdv);
		if (use_bias) b.value.uniform_(-stdv, stdv);
		// register parameter
		for (auto param : {&W, &b}) this->parameters.push_back(param);
	}

	// Linear, !row_major ? Wx + b -> y : xW + b -> y
	// XXX: support >2D tensor as long as the row_major attrib is correct.
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		// output += GEMM(W, X)
		Tensor<Dtype>::gemm(false, false, 1., &W.value, &input.value, 0., &output.value);
		// output += expand(b)
		if (use_bias) {
			size_t batchsize = input.value.getSize(1);
			auto bb = b.value.expand(batchsize);
			output.value += *bb;
			delete bb;
		}
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		if (!output.requires_grad) return;
		// grad of W: g x x^T
		Tensor<Dtype>::gemm(false, true, 1., &output.gradient, &input.value, 0., &W.gradient);
		// grad of X: W^T x g
		if (input.requires_grad) {
			Tensor<Dtype>::gemm(true, false, 1., &W.value, &output.gradient, 0., &input.gradient);
		}
		// grad of b: unexpand(g)
		if (use_bias) {
			auto gb = output.gradient.unexpand(1);
			b.gradient += *gb;
			delete gb;
		}
	}

	// (DEBUG)
	void dumpstat() {
		cout << "  > LinearLayer:" << endl;
		cout << "    * W size " << W.value.getSize() << "\tsum " << W.value.sum() << "\tasum " << W.value.asum();
		cout << "\t | gradW sum " << W.gradient.sum() << "\tasum " << W.gradient.asum() << endl;
		cout << "    * b size " << b.value.getSize() << "\tsum " << b.value.sum() << "\tasum " << b.value.asum();
		cout << "\t | gradb sum " << b.gradient.sum() << "\tasum " << b.gradient.asum() << endl;
	}
};
//ut                                                               linear layer
//>                                                                  // prepare
//>                           Blob<double> X (4, 5); // sample=10, inputSize=12
//>                                                             X.setName("X");
//>                                                            X.value.rand_();
//>                                                        X.dump(true, false);
//>                                                   Blob<double> yhat (2, 5);
//>                                                       yhat.setName("yhat");
//>                                             LinearLayer<double> fc1 (2, 4);
//>                                                    fc1.W.dump(true, false);
//>                                                    fc1.b.dump(true, false);
//>                                                                  // forward
//>                                                       fc1.forward(X, yhat);
//>                                                    yhat.gradient.fill_(1.);
//>                                                                yhat.dump();
//>                                                                 // backward
//>                                                      fc1.backward(X, yhat);
//>                                                               fc1.W.dump();
//>                                                               fc1.b.dump();
//>                                                                   X.dump();
//>                                                                   // update
//>                                                              fc1.SGD(1e-3);
//>                                                             // without bias
//>                                      LinearLayer<double> fc2 (2, 4, false);
//>                                                       fc2.forward(X, yhat);
//>                                                      fc2.backward(X, yhat);

template <typename Dtype>
class Conv2dLayer : public Layer<Dtype> {
public:
	Blob<Dtype> K;
	Blob<Dtype> b;
	bool use_bias = true;

	// We assume that you have already correctly setup the input and
	// output blob size.
	size_t N_, C_, H_, W_, O_, R_, HH_, WW_;

	Conv2dLayer(size_t N, size_t C, size_t H, size_t W, size_t O, size_t R, bool use_bias=true) {
		// O: num of feature maps, C: num of channels
		// R: receptive field (convolution kernel size)
		N_ = N; C_ = C; H_ = H; W_ = W; O_ = O; R_ = R; HH_ = H-R+1; WW_ = W-R+1;
		this->use_bias = use_bias;
		// setup this layer
		K.resize(O, C, R, R);
		if (use_bias) b.resize(O, H-R+1, W-R+1);
		K.setName("Conv2dLayer/K");
		if (use_bias) b.setName("Conv2dLayer/b");
		K.gradient.zero_();
		if (use_bias) b.gradient.zero_();
		// @ref torch:SpatialConvolution::reset
		double stdv = 1. / sqrt(C*R*R);
		K.value.uniform_(-stdv, stdv);
		if (use_bias) b.value.uniform_(-stdv, stdv);
		// register parameters
		for (auto param : {&K, &b}) this->parameters.push_back(param);
	}

	// X(N,C,H,W) *(valid) K(O,C,R,R) (accumulate)-> Y(N,O,H',W')
	//   where H'=H-R+1, W'=W-R+1
	// pseudo code:
	// for k in batch size
	//   for j in output feature map
	//     for i in channel
	//       X(N_k,C_i,H,W) *(valid) K(O_j,C_i,R,R) (accumulate)-> Y(N_k,O_j,H',W')
	//       add bias
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		output.value.zero_();
		for (size_t k = 0; k < N_; k++) { // for k in numbatch
		for (size_t j = 0; j < O_; j++) { // for j in numfeaturemaps
		for (size_t i = 0; i < C_; i++) { // for i in channels
			// X(k,i,:,:) *(valid) K(j,i,:,:) (accumulate)-> Y(k,j,:,:)
			Conv2D("valid",
input.value.data  + (k*C_*H_*W_)   + (i*H_*W_),   H_, W_,
K.value.data      + (j*C_*R_*R_)   + (i*R_*R_),   R_,
output.value.data + (k*O_*HH_*WW_) + (j*HH_*WW_)
			);
		}
		}
		// for each k in numbatch add bias
		llas::axpy(O_*HH_*WW_, (Dtype)1., b.value.data, 1, output.value.data + (k*O_*HH_*WW_), 1);
		}
	}

	// [gK] gradient of K, [gX] gradient of X
	// for k in batchsize
	//   for j in num feature map
	//     for i in channel
	//       X(N_k,C_i,H,W) *(valid) gY(N_k,O_j,R,R) (accumulate)-> gK(O_j,C_i,R,R)
	//       gY(N_k,O_j,H',W') *(full) rot180:K(O_j,C_i,R,R) (accumulate)-> gX(N_k,C_i,H,W)
	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		input.gradient.zero_();
		Tensor<Dtype>* rotK = K.value.rot180();
		for (size_t k = 0; k < N_; k++) { // for k in batch size
		for (size_t j = 0; j < O_; j++) { // for j in num feature map
		for (size_t i = 0; i < C_; i++) { // for i in channel
			// gradient of K(O,C,R,R)
			// X(k,i,:,:) *(valid) gY(k,j,:,:) (accumulate)-> gK(j,i,:,:)
			Conv2D("valid",
input.value.data     + (k*C_*H_*W_)   + (i*H_*W_),    H_, W_,
output.gradient.data + (k*O_*HH_*WW_) + (j*HH_*WW_),  HH_, // FIXME:API: replace HH_ with RH_, RW_, break when non-square kernel/image
K.gradient.data      + (j*C_*R_*R_)   + (i*R_*R_)
			);
			// gradient of X(N,C,H,W)
			// gY(k,j,:,:) *(full) rot180(K(j,i,:,:)) -> gX(k,i,:,:)
			if (input.requires_grad) Conv2D("full",
output.gradient.data + (k*O_*HH_*WW_) + (j*HH_*WW_),  HH_, WW_,
rotK->data           + (j*C_*R_*R_)   + (i*R_*R_)  ,  R_,
input.gradient.data  + (k*C_*H_*W_)   + (i*H_*W_)
			);
		}
		}
		// for each k unexpand the gradient of bias
		llas::axpy(O_*HH_*WW_, (Dtype)1., output.gradient.data + (k*O_*HH_*WW_), 1, b.gradient.data, 1);
		}
		delete rotK;
	}

	// (DEBUG)
	void dumpstat() {
		cout << "  > Conv2dLayer:" << endl;
		cout << "    * K size " << K.value.getSize() << "\tsum " << K.value.sum() << "\tasum " << K.value.asum();
		cout << "\t | gradW sum " << K.gradient.sum() << "\tasum " << K.gradient.asum() << endl;
		cout << "    * b size " << b.value.getSize() << "\tsum " << b.value.sum() << "\tasum " << b.value.asum();
		cout << "\t | gradb sum " << b.gradient.sum() << "\tasum " << b.gradient.asum() << endl;
	}
};
//ut                                                          convolution layer
//>                                           Blob<double> X (2, 3, 5, 5, "X");
//>                                                            X.value.rand_();
//>                               Conv2dLayer<double> conv1 (2, 3, 5, 5, 2, 3);
//>                                           Blob<double> Y (2, 2, 3, 3, "Y");
//>                                                         conv1.forward(X,Y);
//>                                                        X.dump(true, false);
//>                                                        Y.dump(true, false);
//>                                                       Y.gradient.fill_(1.);
//>                                                        conv1.backward(X,Y);
//>                                                                   X.dump();
//>                                                                   Y.dump();
//>                                                             conv1.K.dump();
//>                                                                            
//>                                          Blob<double> image (3, 1, 28, 28);
//>                                          Blob<double> conv2 (3, 3, 24, 24);
//>                            Conv2dLayer<double> lconv2 (3, 1, 28, 28, 3, 5);
//>                                               lconv2.forward(image, conv2);
//>                                              lconv2.backward(image, conv2);

template <typename Dtype>
class ReluLayer : public Layer<Dtype> {
public:
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		auto relu = [](Dtype x) { return x > (Dtype)0. ? x : (Dtype)0.; };
#if defined(USE_OPENMP)
		#pragma omp parallel for
#endif
		for (size_t i = 0; i < input.value.getSize(); i++)
			*output.value.at(i) = relu(*input.value.at(i));
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
#if defined(USE_OPENMP)
		#pragma omp parallel for
#endif
		for (size_t i = 0; i < input.gradient.getSize(); i++)
			*input.gradient.at(i) = (*input.value.at(i) > (Dtype)0.)
				? *output.gradient.at(i) : (Dtype)0.;
	}
};
//ut                                                                 relu layer
//>                                                     Blob<double> X (5, 10);
//>                                                             X.setName("X");
//>                                                 X.value.rand_()->add_(-.5);
//>                                                       X.gradient.fill_(1.);
//>                                                                   X.dump();
//>                                                    ReluLayer<double> relu1;
//>                                                        relu1.forward(X, X);
//>                                                       relu1.backward(X, X);
//>                                                                   X.dump();

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
public:
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		// input.exp().sum(0), sum in the first row
		Tensor<Dtype>* expx = input.value.clone();
		for (size_t j = 0; j < expx->getSize(1); j++) {
			// find maxval of this colomn
			Dtype maxval = *expx->at(0, j);
			for (size_t i = 0; i < expx->getSize(0); i++)
				if (maxval < *expx->at(i,j)) maxval = *expx->at(i,j);
			// subtract the maxval from this column
			for (size_t i = 0; i < expx->getSize(0); i++)
				*expx->at(i,j) -= maxval;
		}
		expx->exp_();
		// save the exp(x_ij) result to output
		output.value.copy(expx->data, output.value.getSize());
		// sum up each column
		for (size_t i = 1; i < expx->getSize(0); i++)
			for (size_t j = 0; j < expx->getSize(1); j++)
				*expx->at(0, j) += *expx->at(i, j);
		// output
		for (size_t i = 0; i < expx->getSize(0); i++)
			for (size_t j = 0; j < expx->getSize(1); j++)
				*output.value.at(i, j) /= (Dtype)1e-7 + *expx->at(0, j);
		delete expx;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		for (size_t sample = 0; sample < input.gradient.getSize(1); sample++) {
			for (size_t row = 0; row < input.gradient.getSize(0); row++) {
				Dtype element = 0.;
				for (size_t k = 0; k < output.gradient.getSize(0); k++) {
					element -= (*output.gradient.at(k, sample))
						* (*output.value.at(row, sample))
						* (*output.value.at(k,sample));
					if (k == row)
						element += (*output.gradient.at(k, sample))
							* (*output.value.at(row,sample));
				}
				*input.gradient.at(row, sample) = element;
			}
		}
	}
};
//ut                                                              softmax layer
//>                                                      Blob<double> x (5, 2);
//>                                                             x.setName("x");
//>                                                            x.value.rand_();
//>                                                      Blob<double> y (5, 2);
//>                                                             y.setName("y");
//>                                                   SoftmaxLayer<double> sm1;
//>                                                          sm1.forward(x, y);
//>                                                       y.gradient.fill_(1.);
//>                                                         sm1.backward(x, y);
//>                                                                   x.dump();
//>                                                                   y.dump();
//>                                                         y.gradient.rand_();
//>                                                         sm1.backward(x, y);
//>                                                                   x.dump();
//>                                                                   y.dump();

template <typename Dtype>
class MSELoss : public Layer<Dtype> {
public:
	double lossval = 0.;
	double MAE = 0.;

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		// (1,batchsize), will do for input
		// (1,batchsize), (batchsize,), will do for label
		assert(input.value.getSize() == label.value.getSize());
		lossval = label.value.MSE(&input.value);
		MAE = label.value.MAE(&input.value);
		*output.value.at(0) = lossval;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		// gX = (2/batchsize) * (yhat - y)
		size_t numsamples = input.value.getSize(1);
		input.gradient.zero_();
		Tensor<Dtype>::axpy(1., &input.value, &input.gradient);
		Tensor<Dtype>::axpy(-1., &label.value, &input.gradient);
		input.gradient.scal_(2./numsamples);
	}

	void report() {
		std::cout << " * MSELoss: " << lossval << "\t(MAE " << MAE << ")" << std::endl;
	}
};
//ut                                                                  MSE layer
//>                                                     Blob<double> y (10, 1);
//>                                                             y.setName("y");
//>                                                          y.value.fill_(0.);
//>                                                   Blob<double> yhat(10, 1);
//>                                                       yhat.setName("yhat");
//>                                                       yhat.value.fill_(1.);
//>                                                      Blob<double> loss (1);
//>                                                      MSELoss<double> loss1;
//>                                               loss1.forward(yhat, loss, y);
//>                                                             loss1.report();
//>                                              loss1.backward(yhat, loss, y);
//>                                                                   y.dump();
//>                                                                yhat.dump();

template <typename Dtype>
class ClassNLLLoss : public Layer<Dtype> {
public:
	double lossval = 0.;

	bool _checksize(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		if (label.value.getDim() == 1) {
			if (input.value.shape[1] != label.value.getSize()) return false;
		} else if (label.value.getDim() == 2) {
			if (input.value.shape[1] != label.value.shape[1]) return false;
		}
		return true;
	}

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		assert(true == _checksize(input, output, label));
		lossval = 0.;
		size_t samples = input.value.getSize(1);
		for (size_t i = 0; i < samples; i++)
			lossval += - log(1e-7 + *input.value.at((size_t)*label.value.at(i), i));
		lossval /= samples;
		*output.value.at(0) = (Dtype)lossval;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		assert(true == _checksize(input, output, label));
		input.gradient.zero_();
		size_t samples = input.value.getSize(1);
		for (size_t i = 0; i < samples; i++)
			*input.gradient.at(*label.value.at(i), i) =
				- 1. / (1e-7 + *input.value.at((size_t)*label.value.at(i), i));
	}

	void report() {
		std::cout << " * ClassNLLLoss: " << lossval << std::endl;
	}
};
//ut                                                         classnllloss layer
//>                                                   SoftmaxLayer<double> sm1;
//>                                                   Blob<double> yhat (5, 2);
//>                                                       yhat.setName("yhat");
//>                                                         yhat.value.rand_();
//>                                                    sm1.forward(yhat, yhat);
//>                                           Blob<double> y (1, 2, "", false);
//>                                                             y.setName("y");
//>                                                          y.value.fill_(1.);
//>                                                      Blob<double> loss (1);
//>                                                 ClassNLLLoss<double> loss1;
//>                                               loss1.forward(yhat, loss, y);
//>                                                             loss1.report();
//>                                              loss1.backward(yhat, loss, y);
//>                                                                   y.dump();
//>                                                                yhat.dump();

template <typename Dtype>
class ClassAccuracy : public Layer<Dtype> {
public:
	double accuracy = 0.;
	size_t numsamples = 0;
	size_t numcorrect = 0;
	size_t numclass = 0;
	Tensor<Dtype> pred;
	Tensor<Dtype> prob;

	ClassAccuracy(void) {
		pred.setName("pred");
		prob.setName("prob");
	}

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		numsamples = input.value.getSize(1);
		numclass   = input.value.getSize(0);
		numcorrect = 0;
		pred.resize(numsamples);
		prob.resize(numsamples);
		for (size_t j = 0; j < numsamples; j++) {
			for (size_t i = 0; i < numclass; i++) {
				if (*prob.at(j) < *input.value.at(i, j)) {
					*prob.at(j) = *input.value.at(i, j);
					*pred.at(j) = i;
				}
			}
		}
		for (size_t i = 0; i < numsamples; i++)
			if ((int)*label.value.at(i) == (int)*pred.at(i)) numcorrect++;
		accuracy = (double)numcorrect / numsamples;
		*output.value.at(0) = accuracy;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		// stub
	}

	void report(bool verbose=false) {
		std::cout << " * Accuracy: " << accuracy << " (" << numcorrect << "/" << numsamples << ")" << std::endl;
		if (verbose) pred.dump();
	}
};
//ut                                                              classaccuracy
//>                                                 ClassAccuracy<double> acc1;
//>                                                 Blob<double> yhat1 (5, 10);
//>                                                 Blob<double> yhat2 (5, 10);
//>                                      Blob<double> y     (1, 10, "", false);
//>                                                          y.value.fill_(1.);
//>                                                             y.setName("y");
//>                                                      yhat1.value.fill_(0.);
//>                                                     yhat1.setName("yhat1");
//>                                                        yhat2.value.rand_();
//>                                                     yhat2.setName("yhat2");
//>                                                       Blob<double> acc (1);
//>                                                                            
//>                                                        y.dump(true, false);
//>                                                    yhat1.dump(true, false);
//>                                                acc1.forward(yhat1, acc, y);
//>                                                              acc1.report();
//>                                                                            
//>                                                        y.dump(true, false);
//>                                                    yhat2.dump(true, false);
//>                                                acc1.forward(yhat2, acc, y);
//>                                                              acc1.report();

template <typename Dtype>
class MaxpoolLayer : public Layer<Dtype> {
public:
	Tensor<int> mask;
	// We assume that the user has correctly setup the blobs
	size_t N_, C_, H_, W_, K_, S_;
	size_t dH_, dW_;

	// FIXME: support padding P
	// X(N,C,H,W), K pooling size, S stride
	MaxpoolLayer(size_t N, size_t C, size_t H, size_t W, size_t K, size_t S) {
		N_ = N; C_ = C; H_ = H; W_ = W; K_ = K; S_ = S;
		dH_ = ceil((H-K+1)/(float)S); dW_ = ceil((W-K+1)/(float)S);
		mask.resize(N,C, dH_, dW_);
		mask.setName("MaxPooling.mask");
	}

	// X(N,C,H,W) (maxpool)-> Y(N,C,ceil(H/K),ceil(W/K))
	// and update the mask
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		for (size_t k = 0; k < N_; k++) { // for k in output batchsize
		for (size_t i = 0; i < C_; i++) { // for i in output channel
		for (size_t h = 0; h < dH_; h++) { // for h in out height
		for (size_t w = 0; w < dW_; w++) { // for w in out width
			Dtype Yhw = -DBL_MAX; // FIXME: double only
			for (size_t ik = 0; ik < K_; ik++) { // for ik in kernel size
			for (size_t jk = 0; jk < K_; jk++) { // for jk in kernel size
Dtype t = *(input.value.data + (k*C_*H_*W_) + (i*H_*W_) + ((h*S_+ik)*W_) + (w*S_+jk));
if (t > Yhw) {
	Yhw = t;
	*mask.at(k, i, h, w) = ik*K_ + jk;
}
			}
			}
*(output.value.data + (k*C_*dH_*dW_) + (i*dH_*dW_) + h*dW_ + w) = Yhw;
		}
		}
		}
		}
	}

	// backward with the mask
	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		for (size_t k = 0; k < N_; k++) { // for k in output batchsize
		for (size_t i = 0; i < C_; i++) { // for i in output channel
		for (size_t h = 0; h < dH_; h++) { // for h in out height
		for (size_t w = 0; w < dW_; w++) { // for w in out width
			Dtype grad   = *output.gradient.at(k, i, h, w);
			int branch = *mask.at(k, i, h, w);
			int kjoff = branch%K_, kioff = (int)branch/K_;
			*input.gradient.at(k, i, h*S_+kioff, w*S_+kjoff) = grad;
		}
		}
		}
		}
	}
};
//ut                                                              maxpool layer
//>               Blob<double> X (1, 2, 6, 6); X.setName("X"); X.value.rand_();
//>                              MaxpoolLayer<double> pool1 (1, 2, 6, 6, 2, 2);
//>               Blob<double> Y (1, 2, 3, 3); Y.setName("Y"); Y.value.rand_();
//>                                                        pool1.forward(X, Y);
//>                                                        X.dump(true, false);
//>                                                        Y.dump(true, false);
//>                                                          pool1.mask.dump();
//>                                                       Y.gradient.fill_(1.);
//>                                                        pool1.backward(X,Y);
//>                                                        Y.dump(false, true);
//>                                                        X.dump(false, true);
//
//ut                                             max pool layer big feature map
//>                Blob<double> X (2,2,10,10); X.setName("X"); X.value.rand_();
//>             Blob<double> Y (2,2,5,5);   Y.setName("Y"); Y.gradient.rand_();
//>                                     MaxpoolLayer<double> p (2,2,10,10,2,2);
//>                                            p.forward(X,Y); p.backward(X,Y);
//>                    X.dump(true, false); p.mask.dump(); Y.dump(true, false);
//>                    Y.dump(false, true); p.mask.dump(); X.dump(false, true);

template <typename Dtype>
class TransposeLayer : public Layer<Dtype> {
public:
	// (M,N) -> (N,M) where N is batchsize
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		//input.value.transpose_(false); // inplace transpose
		//output.value.copy(input.value.data, input.value.getSize());
        	auto xT = input.value.clone(); xT->transpose_();
        	output.value.copy(xT->data, xT->getSize());
		delete xT;

	}
	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		if (input.requires_grad) {
			Tensor<Dtype>* gT = output.gradient.transpose();
			input.gradient.copy(gT->data, gT->getSize());
			delete gT;
		}
	}
};
//ut                                                            transpose layer
//>                                Blob<double> x (2, 5, "x"); x.value.rand_();
//>                             Blob<double> y (5, 2, "y"); y.gradient.rand_();
//>                                              TransposeLayer<double> trans1;
//>                                                       trans1.forward(x, y);
//>                                                      trans1.backward(x, y);
//>                                                         x.dump(); y.dump();

#endif // _LEICHT_LAYER_HPP

// [[[ Benchmarks ]]]

//ut                    linear layer benchmark, 512x512, (fw,bw,up)x1 iteration
//>                                  Blob<double> X (512,512); X.value.rand_();
//>                               Blob<double> Y (512,512); Y.gradient.rand_();
//>                                          LinearLayer<double> fc1 (512,512);
//>                                                                      tic();
//>                                                          fc1.forward(X, Y);
//>                                                         fc1.backward(X, Y);
//>                                                           fc1.update(1e-3);
//>                                                                      toc();
