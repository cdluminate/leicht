/* tensor.cc for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#if !defined(_LEICHT_BLOB_HPP)
#define _LEICHT_BLOB_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include "tensor.hpp"

using namespace std;

template <typename Dtype>
class Blob {
public:
	// holds the value tensor
	Tensor<Dtype> value = Tensor<Dtype>();
	// holds the gradient tensor of value
	Tensor<Dtype> gradient = Tensor<Dtype>();
	// is gradient needed for this blob? true by default
	bool requires_grad = true;
	// optional name
	string name;

	// empty blob constructor
	Blob(){}

	// 1D blob constructor
	Blob(size_t length, string name="", bool requires_grad=true) {
		this->value.resize(length);
		this->requires_grad = requires_grad;
		if (requires_grad) this->gradient.resize(length);
		this->setName(name);
	}

	// 2D blob constructor
	Blob(size_t row, size_t col, string name="", bool requires_grad=true) {
		this->value.resize(row, col);
		this->requires_grad = requires_grad;
		if (requires_grad) this->gradient.resize(row, col);
		this->setName(name);
	}
//ut                                                 blob construction and name
//>                            Blob<double> databatch(12, 10); // d=12, batch10
//>                                             databatch.setName("databatch");
//>                                                           databatch.dump();
//>                            Blob<double> databatchnograd(12, 10, "", false);
//>                                 databatchnograd.setName("databatchnograd");
//>                                                     databatchnograd.dump();
//ut                                                            blob new delete
//>                                       auto x = new Blob<double> (100, 100);
//>                                                                   delete x;

	// Constructor: 3D blob
	Blob(size_t c, size_t h, size_t w, string name="", bool requires_grad=true) {
		this->value.resize(c, h, w);
		this->requires_grad = requires_grad;
		if (requires_grad) this->gradient.resize(c, h, w);
		this->setName(name);
	}

	// Constructor: 4D blob
	Blob(size_t t, size_t c, size_t h, size_t w, string name="", bool requires_grad=true) {
		this->value.resize(t, c, h, w);
		this->requires_grad = requires_grad;
		if (requires_grad) this->gradient.resize(t, c, h, w);
		this->setName(name);
	}

	// Constructor: *D blob
	Blob(std::vector<size_t> shape, string name="", bool requires_grad=true) {
		this->value.resize(shape);
		this->requires_grad = requires_grad;
		if (requires_grad) this->gradient.resize(shape);
		this->setName(name);
	}

	// 1D blob resizer
	void resize(size_t length) {
		value.resize(length);
		if (requires_grad) gradient.resize(length);
	}

	// 2D blob resizer
	void resize(size_t row, size_t col) {
		value.resize(row, col);
		if (requires_grad) gradient.resize(row, col);
	}
//ut                                                                blob resize
//>                                               auto x = new Blob<double> ();
//>                                                              x->resize(10);
//>                                                             x->resize(100);
//>                                                           x->resize(20,10);
//>                                                               x->resize(1);
//>                                                                   delete x;

	// Resizer: resize from *D to 3D
	void resize(size_t c, size_t h, size_t w) {
		value.resize(c, h, w);
		if (requires_grad) gradient.resize(c, h, w);
	}

	// Resizer: resize from *D to 4D
	void resize(size_t t, size_t c, size_t h, size_t w) {
		value.resize(t, c, h, w);
		if (requires_grad) gradient.resize(t, c, h, w);
	}

	// nD blob resizer by std::vector shape
	void resize(std::vector<size_t> shape) {
		value.resize(shape);
		if (requires_grad) gradient.resize(shape);
	}

	// transpose, pseudo-inplace
	void transpose_() {
		assert(value.getDim() == 2);
		Tensor<Dtype>* valueT = value.transpose();
		value.resize(value.shape[1], value.shape[0]);
		value.copy(valueT->data, value.getSize());
		delete valueT;
		if (requires_grad) {
			Tensor<Dtype>* gradientT = gradient.transpose();
			gradient.resize(gradient.shape[1], gradient.shape[0]);
			gradient.copy(gradientT->data, gradient.getSize());
			delete gradientT;
		}
	}
//ut                                                             blob transpose
//>                                                    Blob<double> x (10, 10);
//>                                                            x.value.rand_();
//>                                                         x.gradient.rand_();
//>                                                                   x.dump();
//>                                                             x.transpose_();
//>                                                                   x.dump();


	// blob clone, XXX: don't forget to delete
	Blob<Dtype>* clone() {
		auto newblob = new Blob<Dtype>();
		newblob->name = name;
		newblob->requires_grad = requires_grad;
		newblob->value.resizeAs(&value);
		newblob->gradient.resizeAs(&gradient);
		newblob->value.copy(value.data, value.getSize());
		newblob->gradient.copy(gradient.data, gradient.getSize());
		return newblob;
	}
//ut                                                                 blob clone
//>                                                    Blob<double> x (10, 10);
//>                                                             x.setName("x");
//>                                                            x.value.rand_();
//>                                                    x.gradient.fill_(0.123);
//>                                                                   x.dump();
//>                                                Blob<double>* y = x.clone();
//>                                                            y->setName("y");
//>                                                        y->value.scal_(2.0);
//>                                                     y->gradient.scal_(2.0);
//>                                                                  y->dump();
//>                                                                   delete y;

	// zero gradient
	void zeroGrad() {
		if (requires_grad) this->gradient.zero_();
	}

	// dumper
	void dump() {
		this->value.dump();
		this->gradient.dump();
	}

	// dumper, with flags
	void dump(bool pv, bool pg) {
		if (pv) this->value.dump();
		if (pg) this->gradient.dump();
	}

	// setting name
	void setName(string name) {
		this->name = name;
		this->value.name = name + ".value";
		this->gradient.name = name + ".gradient";
	}

	// compare size of two blobs
	bool sameSize(Blob<Dtype>* x) {
		return value.sameSize(&x->value);
	}

	// get Shape
	void checkShape() {
		if (requires_grad) {
			assert(value.shape.size() == gradient.shape.size());
			for (size_t i = 0; i < value.shape.size(); i++)
				assert(value.shape[i] = gradient.shape[i]);
		}
	}
};

#endif // _LEICHT_BLOB_HPP
