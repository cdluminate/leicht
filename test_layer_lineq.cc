/* tensor.cc for LITE
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#include <iostream>
#include "tensor.hpp"
#include "blob.hpp"
#include "layer.hpp"

using namespace std;

int
main(void)
{
	// AB = C, given B and C, find A
	double b[16] = {1.,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1}; // 4x4
	double c[4]  = {-4.,      -2,        2,        4};       // 1x4

	cout << "Initialize Test Net" << endl;

	Blob<double> X (4, 4, "", false); X.setName("X"); X.value.copy(b, 16);
	Blob<double> y (1, 4, "", false); y.setName("y"); y.value.copy(c, 4);

	Blob<double> yhat (1, 4); yhat.setName("yhat");
	Blob<double> loss (1);    loss.setName("loss");

	LinearLayer<double> fc1 (1, 4, false);
	MSELoss<double> loss1;

	X.dump(true, false);
	y.dump(true, false);
	fc1.W.dump(true, false);
	fc1.b.dump(true, false);

	for (int iteration = 0; iteration < 50; iteration++) {
		cout << ">> Iteration :: " << iteration << endl;
		// -- forward
		fc1.forward(X, yhat);
		loss1.forward(yhat, loss, y);
		// -- zerograd
		yhat.zeroGrad();
		loss.zeroGrad();
		fc1.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, y);
		fc1.backward(X, yhat);
		// -- report
		loss1.report();
		//yhat.dump();
		//fc1.W.dump(false, true);
		//fc1.b.dump(false, true);

		// update
		fc1.SGD(5e-1);
	}
	fc1.W.dump();
	fc1.b.dump();

	return 0;
}
