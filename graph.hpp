/* graph.cc for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#if !defined(_LEICHT_GRAPH_HPP)
#define _LEICHT_GRAPH_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>

#include "tensor.hpp"
#include "blob.hpp"
#include "layer.hpp"

using namespace std;

template <typename Dtype>
class Graph {
private:
	// Layer forward/backward types
	int _EDGE_LINEAR_              = 1;
	int _EDGE_SOFTMAX_             = 2;
	int _EDGE_CLASSNLLLOSS_        = 3;
	int _EDGE_CLASSACCURACY_       = 4;
	int _EDGE_RELU_                = 5;
	int _EDGE_MSE_                 = 6;
	int _EDGE_IDENTITY_            = 7;
public:
	// Note, the layer (edges) must obey the correct topological order.
	// There is currently no topological sorting.

	/* XXX: How to support a Layer in layer.cc from the graph.cc side?
	 *
	 *  0. Register the forward/backward type in the private: section
	 *  1. extend the addLayer method.
	 *  2. extend the _forwardBackwardPass method.
	 *  3. if the layer ships learnable parameters, extend the _update method.
	 *  4. if the layer needs to report something, extend the _report method.
	 */

	string name; // Name of this graph
	size_t batchsize_; // Batchsize used when adding layers
	std::vector<Blob<Dtype>*> nodes; // Blobs in linear space
	std::vector<Layer<Dtype>*> edges; // Layers in linear space
	std::map<string, Blob<Dtype>*> nodeptr; // blobname 2 blobptr
	std::map<string, Layer<Dtype>*> edgeptr; // layername 2 layerptr
	std::map<string, std::vector<Blob<Dtype>*>> bottoms; // layername 2 bottom list
	std::map<string, std::vector<Blob<Dtype>*>> tops; // layername 2 top list
	std::map<string, int> edgetypes; // layername 2 layer forward/backward type
	Blob<Dtype> entryDataBlob; // entry point of data
	Blob<Dtype> entryLabelBlob; // entry point of label

// FIXME: memory leak!
//	~Graph() {
//		// !! Remove things top-down, instead of bottom-up when building things
//		// remove the layers
//		for (size_t i = 0; i < edges.size(); i++)
//			delete edges[i];
//		// remove the nodes
//		for (size_t i = 2; i < nodes.size(); i++)
//			delete nodes[i];
//	}

	Graph(size_t dimdata, size_t dimlabel, size_t batchsize) {
		// init label blob
		entryLabelBlob.requires_grad = false;
		entryLabelBlob.setName("entryLabelBlob");
		nodes.push_back(&entryLabelBlob);
		nodeptr[entryLabelBlob.name] = &entryLabelBlob;
		// init data blob
		entryDataBlob.requires_grad = false;
		entryDataBlob.setName("entryDataBlob");
		nodes.push_back(&entryDataBlob);
		nodeptr[entryDataBlob.name] = &entryDataBlob;
		// resize blobs
		entryDataBlob.resize(dimdata, batchsize);
		entryLabelBlob.resize(dimlabel, batchsize);
		batchsize_ = batchsize;
	}
//ut                                                             Graph creation
//>                                    Graph<double> g (784, 1, 100); g.dump();

	void setName(string name) {
		this->name = name;
	}

	void dump() {
		// setup helper functions
		auto _blobshape = [&](Blob<Dtype>* blob) {
			blob->checkShape();
			cout << "(";
			if (blob->value.shape.size() == 0) {
				cout << ")";
			} else {
				for (auto i : blob->value.shape) cout << i << ",";
				cout << "\b)";
			}
		};
		auto _bottomtop = [&](string layername) {
			auto cursor_b = bottoms.find(layername);
			auto cursor_t = tops.find(layername);
			assert(cursor_b != bottoms.end());
			assert(cursor_t != tops.end());
			cout << "Bottoms[";
			for (Blob<Dtype>* blob : cursor_b->second)
				cout << "'" << blob->name << "',";
			cout << "\b], Tops[";
			for (Blob<Dtype>* blob : cursor_t->second)
				cout << "'" << blob->name << "',";
			cout << "\b]";
		};
		// Dump Header
		std::cout << "Graph \"" << this->name << "\" {" << std::endl;
		cout << endl;
		// Dump stat
		cout << "  * Stat: " << nodes.size() << " Nodes, " << edges.size() << " Edges." << endl;
		cout << endl;
		// Dump node listing
		cout << "  * Nodes (Blobs) Listing:" << endl;
		for (size_t i = 0; i < nodes.size(); i++) {
			cout << "    " << i+1 << ") \"" << nodes[i]->name << "\" ";
			_blobshape(nodes[i]);
			cout << endl;
		}
		cout << endl;
		// Dump edge listing
		cout << "  * Edges (Layers) Listing:" << endl;
		for (size_t i = 0; i < edges.size(); i++) {
			cout << "    " << i+1 << ") \"" << edges[i]->name << "\" ";
			_bottomtop(edges[i]->name);
			cout << endl;
		}
		cout << endl;
		std::cout << "};" << std::endl;;
	}

	Blob<Dtype>* getBlob(string name, bool failure=false) {
		if (nodeptr.find(name) == nodeptr.end()) {
			if (failure) {
				cout << "getBlob: E: " << name << " ";
				fprintf(stderr, "your requested blob doesn't exist!\n");
				exit(EXIT_FAILURE);
			}
			else return nullptr;
		}
		return nodeptr.find(name)->second;
	}

	Layer<Dtype>* getLayer(string name, bool failure=false) {
		if (edgeptr.find(name) == edgeptr.end()) {
			if (failure) {
				cout << "getLayer: E: " << name << " ";
				fprintf(stderr, "your requested layer doesn't exist!\n");
				exit(EXIT_FAILURE);
			}
			else return nullptr;
		}
		return edgeptr.find(name)->second;
	}

	// <internal> helper
	void
	_addLayer_pre_newtop(string topblob, std::vector<size_t> shape) {
		if (getBlob(topblob) == nullptr) {
			Blob<Dtype>* newtop = new Blob<Dtype> (shape);
			newtop->setName(topblob);
			nodes.push_back(newtop);
			nodeptr[topblob] = newtop;
		}
	}

	// <internal> helper
	void
	_addLayer_post_register(Layer<Dtype>* layer, string name, int type,
			std::vector<Blob<Dtype>*> bottom,
			std::vector<Blob<Dtype>*> top) {
		edgetypes[name] = type;
		edges.push_back(layer); // Note topological order
		edgeptr[name] = layer;
		bottoms[name] = std::vector<Blob<Dtype>*> {bottom};
		tops[name] = std::vector<Blob<Dtype>*> {top};
	}

	// addLayer(name, type, srcblob, desetblob, dimdest)
	// * LinearLayer
	void addLayer(string name, string type,
			string srcblob, string destblob, size_t dimdest) {
		// get the srcblob as bottom
		Blob<Dtype>* bottom = getBlob(srcblob, true);
		// create the top bottom if it doesn't exist
		_addLayer_pre_newtop(destblob, std::vector<size_t>{dimdest, batchsize_});
		Blob<Dtype>* top = getBlob(destblob, true);
		// specific to layer type
		if (type == "Linear") {
			// create the linear layer, setup bottom and top
			Layer<Dtype>* layer = new LinearLayer<Dtype> (dimdest, bottom->value.getSize(0));
			layer->name = name;
			_addLayer_post_register(layer, name, _EDGE_LINEAR_,
					std::vector<Blob<Dtype>*> {bottom},
					std::vector<Blob<Dtype>*> {top});
		} else {
			fprintf(stderr, "What's that??\n");
		}
	}
//ut                                                           Graph + 1*linear
//>                                              Graph<double> g (784, 1, 100);
//>         g.addLayer("fc1", "Linear", "entryDataBlob", "fc1", 128); g.dump();

	// addLayer(name, type, srcblob, destblob)
	// * SoftmaxLayer
	// * ReluLayer
	// * Layer (i.e. Identity)
	void addLayer(string name, string type,
			string srcblob, string destblob) {
		// get the srcblob as bottom
		Blob<Dtype>* bottom = getBlob(srcblob, true);
		// setup top if it doesn't exist
		_addLayer_pre_newtop(destblob, bottom->value.shape);
		Blob<Dtype>* top = getBlob(destblob, true);
		// specific to type
		if (type == "Softmax") {
			Layer<Dtype>* layer = new SoftmaxLayer<Dtype> ();
			layer->name = name;
			_addLayer_post_register(layer, name, _EDGE_SOFTMAX_,
					std::vector<Blob<Dtype>*> {bottom},
					std::vector<Blob<Dtype>*> {top});
		} else if (type == "Relu") {
			Layer<Dtype>* layer = new ReluLayer<Dtype> ();
			layer->name = name;
			_addLayer_post_register(layer, name, _EDGE_RELU_,
					std::vector<Blob<Dtype>*> {bottom},
					std::vector<Blob<Dtype>*> {top});
		} else if (type == "Layer" || type == "EYE") {
			Layer<Dtype>* layer = new Layer<Dtype> ();
			layer->name = name;
			_addLayer_post_register(layer, name, _EDGE_IDENTITY_,
					std::vector<Blob<Dtype>*> {bottom},
					std::vector<Blob<Dtype>*> {top});
		} else {
			fprintf(stderr, "What's that??\n");
		}
	}

	// addLayer(name, type, srcblob, destblob, labelblob)
	// * ClassNLLLoss
	// * ClassAccuracy
	// * MSELoss
	void addLayer(string name, string type,
			string srcblob, string destblob, string labelblob) {
		// get the srcblob and labelblob as bottom
		Blob<Dtype>* bottom = getBlob(srcblob, true);
		Blob<Dtype>* label  = getBlob(labelblob, true);
		// setup top
		_addLayer_pre_newtop(destblob, std::vector<size_t>{1});
		Blob<Dtype>* top = getBlob(destblob, true);
		// specific
		if (type == "ClassNLLLoss") {
			Layer<Dtype>* layer = new ClassNLLLoss<Dtype> ();
			layer->name = name;
			_addLayer_post_register(layer, name, _EDGE_CLASSNLLLOSS_,
					std::vector<Blob<Dtype>*> {bottom, label},
					std::vector<Blob<Dtype>*> {top});
		} else if (type == "ClassAccuracy") {
			Layer<Dtype>* layer = new ClassAccuracy<Dtype> ();
			layer->name = name;
			_addLayer_post_register(layer, name, _EDGE_CLASSACCURACY_,
					std::vector<Blob<Dtype>*> {bottom, label},
					std::vector<Blob<Dtype>*> {top});
		} else if (type == "MSELoss") {
			Layer<Dtype>* layer = new MSELoss<Dtype> ();
			layer->name = name;
			_addLayer_post_register(layer, name, _EDGE_MSE_,
					std::vector<Blob<Dtype>*> {bottom, label},
					std::vector<Blob<Dtype>*> {top});
		} else {
			fprintf(stderr, "What's that??\n");
		}
	}

	void zeroGrad(void) {
		for (Blob<Dtype>* blob : nodes)
			blob->zeroGrad();
		for (Layer<Dtype>* layer : edges)
			layer->zeroGrad();
	}

	void update(double lr, string optim="SGD", bool verbose=false) {
		for (auto iter = edges.begin(); iter != edges.end(); iter++) {
			_update(*iter, lr, optim, verbose);
			if (verbose) {
				cout << "* Update " << "\x1b[31m" << (*iter)->name << "\x1b[m "
					<< "with learning rate \x1b[31m" << lr << "\x1b[m "
					<< endl;
			}
		}
	}

	void forward(bool verbose=false) {
		for (auto iter = edges.begin(); iter != edges.end(); iter++)
			_forwardBackwardPass(*iter, true, verbose);
	}
//ut                                                 Graph + 1*linear + forward
//>                                              Graph<double> g (784, 1, 100);
//>                   g.addLayer("fc1", "Linear", "entryDataBlob", "fc1", 128);
//>                                                                   g.dump();
//>                                                            g.forward(true);
//
//ut                                                 Graph + 2*linear + forward
//>                                              Graph<double> g (784, 1, 100);
//>                   g.addLayer("fc1", "Linear", "entryDataBlob", "fc1", 128);
//>                              g.addLayer("fc2", "Linear", "fc1", "fc2", 10);
//>                                                                   g.dump();
//>                                                            g.forward(true);
//
//ut                                          Graph + 1*linear + 1*sm + forward
//>                                              Graph<double> g (784, 1, 100);
//>                    g.addLayer("fc1", "Linear", "entryDataBlob", "fc1", 10);
//>                                 g.addLayer("sm1", "Softmax", "fc1", "sm1");
//>                                                                   g.dump();
//>                                                            g.forward(true);

	void backward(bool verbose=false) {
		for (auto iter = edges.rbegin(); iter != edges.rend(); iter++)
			_forwardBackwardPass(*iter, false, verbose);
	}
//ut         Graph + 1*linear + 1*sm + 1*classnll + forward + backward + update
//>                                              Graph<double> g (784, 1, 100);
//>                    g.addLayer("fc1", "Linear", "entryDataBlob", "fc1", 10);
//>                                 g.addLayer("sm1", "Softmax", "fc1", "sm1");
//>        g.addLayer("cls1", "ClassNLLLoss", "sm1", "cls1", "entryLabelBlob");
//>                                                                   g.dump();
//>                                                            g.forward(true);
//>                                                           g.backward(true);
//>                                                g.update(1e-1, "SGD", true);

	void report(bool verbose=false) {
		for (auto iter = edges.begin(); iter != edges.end(); iter++)
			_report(*iter, verbose);
	}

	void _update(Layer<Dtype>* cursor, double lr, string optim="SGD", bool verbose=false) {
		int edgetype = edgetypes.find(cursor->name)->second;
		if (edgetype == _EDGE_LINEAR_) {
			LinearLayer<Dtype>* layer = (LinearLayer<Dtype>*)&*cursor;
			// update!
			layer->update(lr, optim);
		}
		if (verbose) cout << "* Update " << "\x1b[31m" << cursor->name << "\x1b[m "
			<< "with learning rate \x1b[31m" << lr << "\x1b[m "
			<< endl;
	}

	void _report(Layer<Dtype>* cursor, bool verbose=false) {
		int edgetype = edgetypes.find(cursor->name)->second;
		if (edgetype == _EDGE_CLASSNLLLOSS_) {
			ClassNLLLoss<Dtype>* layer = (ClassNLLLoss<Dtype>*)&*cursor;
			// report!
			layer->report();
		} else if (edgetype == _EDGE_CLASSACCURACY_) {
			ClassAccuracy<Dtype>* layer = (ClassAccuracy<Dtype>*)&*cursor;
			// report!
			layer->report();
		} else if (edgetype == _EDGE_MSE_) {
			((MSELoss<Dtype>*)&*cursor)->report();
		}
		if (verbose) cout << "* Report "
			<< "\x1b[31m" << cursor->name << "\x1b[m "
			<< endl;
	}

	void _forwardBackwardPass(Layer<Dtype>* cursor, bool isforward=true,
			bool verbose=false) {
		// lookup layer type
		int edgetype = edgetypes.find(cursor->name)->second;
		// get bottom and top
		auto bottom = bottoms.find(cursor->name)->second;
		auto top = tops.find(cursor->name)->second;
		// forward / backward depending on type
		if (edgetype == _EDGE_LINEAR_) {
			auto layer = (LinearLayer<Dtype>*)&*cursor;
			if (isforward) layer->forward(*bottom[0], *top[0]);
			else layer->backward(*bottom[0], *top[0]);
		} else if (edgetype == _EDGE_SOFTMAX_) {
			auto layer = (SoftmaxLayer<Dtype>*)&*cursor;
			if (isforward) layer->forward(*bottom[0], *top[0]);
			else layer->backward(*bottom[0], *top[0]);
		} else if (edgetype == _EDGE_CLASSNLLLOSS_) {
			auto layer = (ClassNLLLoss<Dtype>*)&*cursor;
			if (isforward) layer->forward(*bottom[0], *top[0], *bottom[1]);
			else layer->backward(*bottom[0], *top[0], *bottom[1]);
		} else if (edgetype == _EDGE_CLASSACCURACY_) {
			auto layer = (ClassAccuracy<Dtype>*)&*cursor;
			if (isforward) layer->forward(*bottom[0], *top[0], *bottom[1]);
			else layer->backward(*bottom[0], *top[0], *bottom[1]);
		} else if (edgetype == _EDGE_RELU_) {
			auto layer = (ReluLayer<Dtype>*)&*cursor;
			if (isforward) layer->forward(*bottom[0], *top[0]);
			else layer->backward(*bottom[0], *top[0]);
		} else if (edgetype == _EDGE_MSE_) {
			auto layer = (MSELoss<Dtype>*)&*cursor;
			if (isforward) layer->forward(*bottom[0], *top[0], *bottom[1]);
			else layer->backward(*bottom[0], *top[0], *bottom[1]);
		} else {
			fprintf(stderr, "E: Graph::forward not implemented for this type!\n");
			exit(EXIT_FAILURE);
		}
		// report if verbose
		if (verbose) cout << "* " << (isforward ? "Forward " : "Backward ")
			<< "\x1b[31m(type " << edgetype << ") " << cursor->name << "\x1b[m ";
		if (verbose && isforward) {
			cout << "from bottom ";
			for (size_t i = 0; i < bottom.size(); i++)
				cout << "\x1b[31m" << bottom[i]->name << "\x1b[m, ";
			cout << "to top ";
			for (size_t i = 0; i < top.size(); i++)
				cout << "\x1b[31m" << top[i]->name << "\x1b[m, ";
			cout << endl;
		} else if (verbose && !isforward) {
			cout << "from top ";
			for (size_t i = 0; i < top.size(); i++)
				cout << "\x1b[31m" << top[i]->name << "\x1b[m, ";
			cout << "to bottom ";
			for (size_t i = 0; i < bottom.size(); i++)
				cout << "\x1b[31m" << bottom[i]->name << "\x1b[m, ";
			cout << endl;
		}
	}
};

#endif // _LEICHT_GRAPH_HPP
