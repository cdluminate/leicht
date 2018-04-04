/* tensor.cc for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#include <iostream>
#include <vector>
#include "leicht.hpp"

using namespace std;

unsigned int batchsize = 64;
double       lr        = 1e-3; // reference lr=1e-3
int          maxiter   = 1000;
int          iepoch    = 37800/batchsize;
int          overfit   = 10; // (DEBUG) let it overfit on howmany batches
int          testevery = 100;
vector<double> validaccuhist;
vector<double> validlosshist;
Curve curve_train_loss;
Curve curve_train_accuracy;

int
main(void)
{
	cout << ">> Reading MNIST dataset" << endl;

	Tensor<double> trainImages (37800, 784);
	trainImages.setName("trainImages");
	leicht_hdf5_read("mnist.th.h5", "/train/images", 0, 0, 37800, 784, trainImages.data);
	Tensor<double> trainLabels (37800, 1);
	trainLabels.setName("trainLabels");
	leicht_hdf5_read("mnist.th.h5", "/train/labels", 0, 0, 37800, 1, trainLabels.data);

	cout << ">> Reading MNIST validation dataset" << endl;

	Tensor<double> valImages(4200, 784); valImages.setName("valImages");
	leicht_hdf5_read("mnist.th.h5", "/val/images", 0, 0, 4200, 784, valImages.data);
	Tensor<double> valLabels(4200, 1);   valLabels.setName("valLabels");
	leicht_hdf5_read("mnist.th.h5", "/val/labels", 0, 0, 4200, 1, valLabels.data);

	cout << ">> Initialize Network" << endl;

	Blob<double> label  (1, batchsize, "label", false);

	Blob<double> image  (batchsize, 784, "image", false);
	Blob<double> imageT (784, batchsize, "imageT", false);
	Blob<double> o1     (256, batchsize);       o1.setName("o1");
	Blob<double> o2     (10, batchsize);
	Blob<double> yhat   (10, batchsize);       yhat.setName("yhat");
	Blob<double> loss   (1);                   loss.setName("loss");
	Blob<double> acc    (1);                   acc.setName("accuracy");

	TransposeLayer<double>trans1;
	LinearLayer<double>   fc1 (256, 784);
	ReluLayer<double>     relu1;
	LinearLayer<double>   fc2 (10, 256);
	SoftmaxLayer<double>  sm1;
	ClassNLLLoss<double>  loss1;
	ClassAccuracy<double> acc1;

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < maxiter; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;

		// -- get batch
		image.value.copy(
trainImages.data + (iteration%iepoch)*batchsize*784, batchsize*784);
		label.value.copy(
trainLabels.data + (iteration%iepoch)*batchsize*1, batchsize*1);
		image.value.scal_(1./255.);

//cout << "image .sum " << image.value.sum() << endl;
//label.dump();

		// -- forward
		trans1.forward(image, imageT);
		fc1.forward(imageT, o1);
		relu1.forward(o1, o1);
		fc2.forward(o1, o2);
		sm1.forward(o2, yhat);
		loss1.forward(yhat, loss, label);
		acc1.forward(yhat, acc, label);
		// -- zerograd
		fc1.zeroGrad();
		fc2.zeroGrad();
		o1.zeroGrad();
		o2.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, label);
		sm1.backward(o2, yhat);
		fc2.backward(o1, o2);
		relu1.backward(o1, o1);
		fc1.backward(imageT, o1);
		// -- report
		loss1.report();
		acc1.report();
		fc1.dumpstat();
		curve_train_loss.append(iteration, loss1.lossval);
		curve_train_accuracy.append(iteration, acc1.accuracy);
		// update
		fc1.update(lr, "SGD");

if ((iteration+1)%testevery==0) {
cout << ">> Validate:" << endl;
vector<double> accuracy;
vector<double> l;
for (int t = 0; t < 42; t++) {
	// -- get batch
	image.value.copy(valImages.data + t*batchsize*784, batchsize*784);
	label.value.copy(valLabels.data + t*batchsize*1, batchsize*1);
	image.value.scal_(1./255.);

	// -- forward
	trans1.forward(image, imageT);
	fc1.forward(imageT, o1);
	relu1.forward(o1, o1);
	fc2.forward(o1, o2);
	sm1.forward(o2, yhat);
	loss1.forward(yhat, loss, label);
	acc1.forward(yhat, acc, label);

	acc1.report();

	accuracy.push_back(*acc.value.at(0));
	l.push_back(*loss.value.at(0));
}
//for (auto i : accuracy) cout << i << " " << endl;
double a = 0; for (auto i : accuracy) a += i; a /= accuracy.size();
cout << "  * Accuracy: " << a << endl;
double b = 0; for (auto i : l) b += i; b /= l.size();
cout << "  * Loss: " << b << endl;
validlosshist.push_back(b);
validaccuhist.push_back(a);
}
	}
	// show history
	for (auto i : validlosshist) cout << i << " "; cout << endl;
	for (auto i : validaccuhist) cout << i << " "; cout << endl;

	curve_train_loss.draw("loss.svg");
	curve_train_accuracy.draw("acc.svg");

	return 0;
}
