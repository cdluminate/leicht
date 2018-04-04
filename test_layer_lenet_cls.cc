/* test_mnist_lenet_cls.cc for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#include <iostream>
#include "leicht.hpp"

using namespace std;

unsigned int batchsize = 64;
double       lr        = 1e-3; // reference lr=1e-3
int          maxiter   = 1000;
int          iepoch    = 37800/batchsize;                                       
int          itepoch   = 4200/batchsize;
int          overfit   = 10; // (DEBUG) let it overfit on howmany batches       
int          testevery = 100;                                                   
string       optim     = "SGDM";
vector<double> validaccuhist;                                                   
vector<double> validlosshist;   
Curve cv_train_loss;
Curve cv_train_acc;
Curve cv_test_loss;
Curve cv_test_acc;

int
main(void)
{
	leicht_threads(2);
	cout << ">> Reading MNIST training dataset" << endl;

	Tensor<double> trainImages(37800, 784); trainImages.setName("trainImages");
	leicht_hdf5_read("mnist.th.h5", "/train/images", 0, 0, 37800, 784, trainImages.data);
	Tensor<double> trainLabels(37800, 1);   trainLabels.setName("trainLabels");
	leicht_hdf5_read("mnist.th.h5", "/train/labels", 0, 0, 37800, 1, trainLabels.data);

	cout << ">> Reading MNIST validation dataset" << endl;

	Tensor<double> valImages(4200, 784); valImages.setName("valImages");
	leicht_hdf5_read("mnist.th.h5", "/val/images", 0, 0, 4200, 784, valImages.data);
	Tensor<double> valLabels(4200, 1);   valLabels.setName("valLabels");
	leicht_hdf5_read("mnist.th.h5", "/val/labels", 0, 0, 4200, 1, valLabels.data);

	cout << ">> Initialize Network" << endl;

	// reference: caffe/examples/mnist/lenet
	Blob<double> label   (1, batchsize, "label", false);
	Blob<double> X       (batchsize, 784, "X", false);
	Blob<double> image   (batchsize, 1, 28, 28, "image", false);
	Blob<double> conv1   (batchsize, 20, 24, 24);             conv1.setName("conv1");
	Blob<double> pool1   (batchsize, 20, 12, 12);             pool1.setName("pool1");
	Blob<double> conv2   (batchsize, 50, 8, 8);               conv2.setName("conv2");
	Blob<double> pool2   (batchsize, 50, 4, 4);               pool2.setName("pool2");
	Blob<double> pool2f  (batchsize, 800);                    pool2f.setName("pool2f");
	Blob<double> pool2fT (800, batchsize);                    pool2fT.setName("pool2fT");
	Blob<double> ip1     (500, batchsize);                    ip1.setName("ip1");
	Blob<double> ip2     (10, batchsize);                     ip2.setName("ip2");
	Blob<double> sm1     (10, batchsize);                     sm1.setName("sm1");
	Blob<double> loss    (1);                                 loss.setName("loss");
	Blob<double> acc     (1);                                 acc.setName("acc");

	Layer<double>         lid1;  // X->image bs,784->bs,1,28,28
	Conv2dLayer<double>   lconv1 (batchsize, 1, 28, 28, 20, 5); // image->conv1 bs,1,28,28->bs,20,24,24
	MaxpoolLayer<double>  lpool1 (batchsize, 20, 24, 24, 2, 2); // conv1->pool1 bs,20,24,24->bs,20,12,12
	Conv2dLayer<double>   lconv2 (batchsize, 20, 12, 12, 50, 5); // pool1->conv2 bs,20,12,12->bs,50,8,8
	MaxpoolLayer<double>  lpool2 (batchsize, 50, 8, 8, 2, 2); // conv2->pool2 bs,50,8,8->bs,50,4,4
	Layer<double>         lid2;  // pool2->pool2f(lattened) bs,50,4,4->bs,800
        TransposeLayer<double>lt1;   // pool2f->pool2fT bs,800->800,bs
	LinearLayer<double>   lfc1   (500, 800); // pool2fT->ip1 800,bs->500,bs
	ReluLayer<double>     lrelu1; // ip1->ip1
	LinearLayer<double>   lfc2   (10, 500); // ip1->ip2 500,bs->10,bs
	SoftmaxLayer<double>  lsm1;  // ip2->sm1
	ClassNLLLoss<double>  lloss; // sm1->loss
	ClassAccuracy<double> lacc;  // sm1->acc

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < maxiter; iteration++) {
		tic();
		leicht_bar_train(iteration);

		// -- get batch
		X.value.copy(
//trainImages.data + (iteration%overfit)*batchsize*784, batchsize*784);
trainImages.data + (iteration%iepoch)*batchsize*784, batchsize*784);
		label.value.copy(
//trainLabels.data + (iteration%overfit)*batchsize*1, batchsize*1);
trainLabels.data + (iteration%iepoch)*batchsize*1, batchsize*1);
		X.value.scal_(1./255.);

		// -- forward : unfold with vim: BEIGN,ENDs/; /;\r/g
		lid1.forward(X, image);             //X.dump(true, false); image.dump(true, false);
		lconv1.forward(image, conv1);       //conv1.dump(true, false);
		lpool1.forward(conv1, pool1);       //pool1.dump(true, false);
		lconv2.forward(pool1, conv2);       //conv2.dump(true, false);
		lpool2.forward(conv2, pool2);       //pool2.dump(true, false);
		lid2.forward(pool2, pool2f);        //pool2f.dump(true, false);
		lt1.forward(pool2f, pool2fT);       //pool2fT.dump(true, false);
		//auto p2T = pool2f.value.transpose();
		//pool2fT.value.copy(p2T->data, p2T->getSize());
		//delete p2T;
		lfc1.forward(pool2fT, ip1);         //ip1.dump(true, false);
		lrelu1.forward(ip1, ip1);           //ip1.dump(true, false);
		lfc2.forward(ip1, ip2);             //ip2.dump(true, false);
		lsm1.forward(ip2, sm1);             //sm1.dump(true, false);
		lloss.forward(sm1, loss, label);    //loss.dump(true, false);
		lacc.forward(sm1, loss, label);     //acc.dump(true, false);

		// -- zerograd
		label.zeroGrad(); X.zeroGrad(); image.zeroGrad();
		conv1.zeroGrad(); pool1.zeroGrad(); conv2.zeroGrad();
		pool2.zeroGrad(); pool2f.zeroGrad(); pool2fT.zeroGrad();
		ip1.zeroGrad(); ip2.zeroGrad(); sm1.zeroGrad();
		loss.zeroGrad(); acc.zeroGrad();

		lid1.zeroGrad(); lconv1.zeroGrad(); lpool1.zeroGrad();
		lconv2.zeroGrad(); lpool2.zeroGrad(); lid2.zeroGrad();
		lfc1.zeroGrad(); lrelu1.zeroGrad(); lfc2.zeroGrad();
		lsm1.zeroGrad(); lloss.zeroGrad(); lacc.zeroGrad();

		// -- backward : unfold with vim: BEIGN,ENDs/; /;\r/g
		lloss.backward(sm1, loss, label);   //sm1.dump();
		lsm1.backward(ip2, sm1);            //ip2.dump();
		lfc2.backward(ip1, ip2);            //ip1.dump();
		lrelu1.backward(ip1, ip1);          //ip1.dump();
		lfc1.backward(pool2fT, ip1);        //pool2fT.dump();
                lt1.backward(pool2f, pool2fT);      //pool2f.dump();
		//auto p2fT = pool2fT.gradient.transpose();
		//pool2f.gradient.copy(p2fT->data, p2fT->getSize());
		// delete p2fT;
		lid2.backward(pool2, pool2f);       //pool2.dump();
	   	lpool2.backward(conv2, pool2);      //conv2.dump();
		lconv2.backward(pool1, conv2);      //pool1.dump();
	   	lpool1.backward(conv1, pool1);      //conv1.dump();
		lconv1.backward(image, conv1);      //image.dump();

		// regularize
		lconv1.regularization(); lconv2.regularization();
		lfc1.regularization();   lfc2.regularization();

		// -- report
		lloss.report(); lacc.report(true);
		label.dump(true, false);
		lconv1.dumpstat(); lconv2.dumpstat();
		lfc1.dumpstat();   lfc2.dumpstat();
		//pool1.dump(true, false);
		
		cv_train_loss.append(iteration, lloss.lossval);
		cv_train_acc.append(iteration, lacc.accuracy);

		// -- update
		lconv1.update(lr, optim); lconv2.update(lr, optim);
		lfc1.update(lr, optim); lfc2.update(lr, optim);

		toc();

		// -- validation
		if (testevery!=0 && iteration%testevery==0) {
			leicht_bar_val(iteration);
			Tensor<double> cvloss (itepoch);
			Tensor<double> cvacc  (itepoch);
			for (int t = 0; t < itepoch; t++) {
				// -- get batch
				X.value.copy(valImages.data + t*batchsize*784, batchsize*784);
				label.value.copy(valLabels.data + t*batchsize*1, batchsize*1);
				X.value.scal_(1./255.);

	            // -- forward : unfold with vim: BEIGN,ENDs/; /;\r/g
        		lid1.forward(X, image);             //X.dump(true, false); image.dump(true, false);
        		lconv1.forward(image, conv1);       //conv1.dump(true, false);
        		lpool1.forward(conv1, pool1);       //pool1.dump(true, false);
        		lconv2.forward(pool1, conv2);       //conv2.dump(true, false);
        		lpool2.forward(conv2, pool2);       //pool2.dump(true, false);
        		lid2.forward(pool2, pool2f);        //pool2f.dump(true, false);
        		lt1.forward(pool2f, pool2fT);       //pool2fT.dump(true, false);
        		//auto p2T = pool2f.value.transpose();
        		//pool2fT.value.copy(p2T->data, p2T->getSize());
        		//delete p2T;
        		lfc1.forward(pool2fT, ip1);         //ip1.dump(true, false);
        		lrelu1.forward(ip1, ip1);           //ip1.dump(true, false);
        		lfc2.forward(ip1, ip2);             //ip2.dump(true, false);
        		lsm1.forward(ip2, sm1);             //sm1.dump(true, false);
        		lloss.forward(sm1, loss, label);    //loss.dump(true, false);
        		lacc.forward(sm1, loss, label);     //acc.dump(true, false);

				// -- report
				//lloss.report(); lacc.report();
				cout << "."; cout.flush();
				cvloss.data[t] = lloss.lossval;
				cvacc.data[t] = lacc.accuracy;
			}
			cout << endl;
			cout << "Test Loss" << cvloss.sum() / cvloss.getSize() << endl;
			cv_test_loss.append(iteration, cvloss.sum() / cvloss.getSize());
			cout << "Test Accu" << cvacc.sum() / cvacc.getSize() << endl;
			cv_test_acc.append(iteration, cvacc.sum() / cvacc.getSize());
		}
	}

	cv_train_loss.draw("lenet-train-loss.svg");
	cv_train_acc.draw("lenet-train-acc.svg");
	cv_test_loss.draw("lenet-test-loss.svg");
	cv_test_acc.draw("lenet-test-acc.svg");

	return 0;
}
