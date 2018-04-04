#include <iostream>
#include "leicht.hpp"

using namespace std;

unsigned int batchsize = 64;
double       lr        = 1e-1; // reference lr=1e-3
int          maxiter   = 1000;
int          iepoch    = 37800/batchsize;
int          overfit   = 10; // (DEBUG) let it overfit on howmany batches
int          testevery = 100;
vector<double> validlosshist;

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

	Blob<double> image  (batchsize, 784, "", false);
	Blob<double> imageT (784, batchsize, "", false); imageT.setName("imageT");
	Blob<double> label (1, batchsize, "", false);    label.setName("label");
	Blob<double> o1 (128, batchsize);                o1.setName("o1");
	Blob<double> o2 (128, batchsize);                o2.setName("o2");
	Blob<double> yhat (1, batchsize);                yhat.setName("yhat");
	Blob<double> loss (1);                     loss.setName("loss");

	TransposeLayer<double>                 trans1;
	LinearLayer<double>                    fc1 (128, 784);
	ReluLayer<double>                      relu1;
	LinearLayer<double>                    fc2 (128, 128);
	ReluLayer<double>                      relu2;
	LinearLayer<double>                    fc3 (1, 128);
	MSELoss<double>                        loss1;

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < maxiter; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;

		// -- get batch
		image.value.copy(
trainImages.data + (iteration%iepoch)*batchsize*784, batchsize*784);
		label.value.copy(
trainLabels.data + (iteration%iepoch)*batchsize*1, batchsize*1);
		image.value.scal_(1./255.);
		
		// -- forward
		trans1.forward(image, imageT);
		fc1.forward(imageT, o1);
		relu1.forward(o1, o1); // inplace relu
		fc2.forward(o1, o2);
		relu2.forward(o2, o2); // inplace relu
		fc3.forward(o2, yhat);
		loss1.forward(yhat, loss, label);
		// -- zerograd
		fc1.zeroGrad();
		fc2.zeroGrad();
		fc3.zeroGrad();
		o1.zeroGrad();
		o2.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, label);
		fc3.backward(o2, yhat);
		relu2.backward(o2, o2);
		fc2.backward(o1, o2);
		relu1.backward(o1, o1); // inplace relu
		fc1.backward(imageT, o1);
		// -- report
		loss1.report();
		label.dump(true, false);
		yhat.dump(true, false);
		//fc1.dumpstat();
		// update
		fc1.update(lr);
		fc2.update(lr);

if ((iteration+1)%testevery==0) {
cout << ">> Validate:" << endl;
vector<double> l;
for (int t = 0; t < 42; t++) {
	// -- get batch
	image.value.copy(valImages.data + t*batchsize*784, batchsize*784);
	label.value.copy(valLabels.data + t*batchsize*1, batchsize*1);
	image.value.scal_(1./255.);

	// -- forward
	trans1.forward(image, imageT);
	fc1.forward(imageT, o1);
	relu1.forward(o1, o1); // inplace relu
	fc2.forward(o1, o2);
	relu2.forward(o2, o2); // inplace relu
	fc3.forward(o2, yhat);
	loss1.forward(yhat, loss, label);

	l.push_back(*loss.value.at(0));
}
//for (auto i : accuracy) cout << i << " " << endl;
double b = 0; for (auto i : l) b += i; b /= l.size();
cout << "  * Loss: " << b << endl;
validlosshist.push_back(b);
}
	}
	// show history
	for (auto i : validlosshist) cout << i << " "; cout << endl;

	return 0;
}
