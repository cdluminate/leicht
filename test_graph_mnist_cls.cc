/* tensor.cc for LITE
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#include <iostream>
#include "leicht.hpp"
using namespace std;

unsigned int   batchsize = 64;
double         lr        = 1e-3; // reference lr=1e-3
int            maxiter   = 1000;
int            iepoch    = 37800/batchsize;
int            overfit   = 10; // (DEBUG) let it overfit on howmany batches
int            testevery = 100;
vector<double> validaccuhist;
vector<double> validlosshist;

int
main(void)
{
	leicht_version();
	cout << ">> Reading MNIST training dataset" << endl;

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

	Graph<double> net (784, 1, 100);
	net.name = "test net";
	net.addLayer("fc1", "Linear", "entryDataBlob", "fc1", 10);
	net.addLayer("sm1", "Softmax", "fc1", "sm1");
	net.addLayer("cls1", "ClassNLLLoss", "sm1", "cls1", "entryLabelBlob");
	net.addLayer("acc1", "ClassAccuracy", "sm1", "acc1", "entryLabelBlob");
	net.dump();

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < maxiter; iteration++) {
		leicht_bar_train(iteration);
		// -- get batch
		Tensor<double>* batchIm = new Tensor<double> (100, 784);
		batchIm->copy(trainImages.data + (iteration%iepoch)*batchsize*784, batchsize*784);
		batchIm->transpose_();
		batchIm->scal_(1./255.);
		net.getBlob("entryDataBlob", true)->value.copy(batchIm->data, 784*batchsize);
		net.getBlob("entryLabelBlob", true)->value.copy(trainLabels.data + (iteration%iepoch)*batchsize*1, batchsize*1);
		delete batchIm;

		// -- forward
		net.forward();
		// -- zerograd
		net.zeroGrad();
		// -- backward
		net.backward();
		// -- report
		net.report();
		// -- update
		net.update(1e-3, "Adam");

		// -- test every
		if ((iteration+1)%testevery==0) {
			leicht_bar_val(iteration);
			vector<double> accuracy;
			vector<double> l;
			for (int t = 0; t < 42; t++) {
				// -- get batch
				Tensor<double>* tbatchIm = new Tensor<double> (100, 784);
				tbatchIm->copy(valImages.data + t*batchsize*784, batchsize*784);
				tbatchIm->transpose_();
				tbatchIm->scal_(1./255.);
				net.getBlob("entryDataBlob", true)->value.copy(tbatchIm->data, 784*batchsize);
				net.getBlob("entryLabelBlob", true)->value.copy(valLabels.data + t*batchsize*1, batchsize*1);
				delete tbatchIm;

				net.forward(); net.report();
			}
		}
	}
	// show history
	for (auto i : validlosshist) cout << i << " "; cout << endl;
	for (auto i : validaccuhist) cout << i << " "; cout << endl;
	return 0;
}
