Leicht
======

Still under development. Stage: alpha5.

A light weight neural network implementation in C++ **from scratch**,
for **educational purpose**, inspired by *Caffe*, *(Lua)Torch* and *Pytorch*.

**Features**
  1. Light Weight and Simple. Only make necessary abstractions.
  2. Almost Dependency Free (from Scratch). Used some I/O libs.
  3. Educational Purpose. Verify one's understanding to neural nets.
  4. Designed in a mixed style of Caffe and (Lua)Torch, (and maybe PyTorch).

"Leicht" is a German word.

Design
------

There are 4 *core* concepts in this framework, which are:
*Tensor*, *Blob*, *Layer*, and *Graph*. Note that the default
vector for this project is column vector.

**Tensor**

  Generalized container of numerical data. Vectors, matrices or any
  higher-dimensional number blocks are regarded as Tensor, where the
  data is stored in a contiguous memory block.

  See ``tensor.hpp`` for detail.

**Blob**

  Combination of two Tensors, one for the value and another for its
  gradient. This is useful in the Caffe-styled computation graph,
  where the backward pass just uses the forward graph instead of
  extending the graph for gradient computation and parameter update.

  See ``blob.hpp`` for detail.

**Layer**

  Network layers, including loss functions. Each of them takes some
  input Blobs and output Blobs as argument during forward and backward.

  See ``layer.hpp`` for detail.

**Graph**

  Graph, or say Directed Acyclic Graph, is the computation graph
  interpretation of the neural network, where the nodes are Blobs,
  the edge (or edge groups) are Layers. The graph is static graph.

  See ``graph.hpp`` for detail.

Apart from Core part, there are some auxiliary components:

**Dataloader**

  Basically an I/O helper, which reads dataset or data batch from disk
  to memory. This is not a key part of the project.

  See ``dataloader.hpp`` for detail.

**Curve**

  Save the curve data to ASCII file, and optionally draw a picture for you.
  Although one can parse the screen output with UNIX blackmagics e.g. ``awk``.

  See ``curve.hpp`` for detail.

Brief Usage
-----------

Just include the header in your C++ file like this

.. code:: cpp

  #include "leicht.hpp"

Example of network definition:

.. code:: cpp

  // create a network(static graph), input dim 784, label dim 1, batch 100
  // There are two pre-defined blobs in the graph: entry{Data,Label}Blob
  Graph<double> net (784, 1, 100);

  // add a layer, name=fc1, type=Linear, bottom=entryDataBlob, top=fc1, out dim=10
  net.addLayer("fc1", "Linear", "entryDataBlob", "fc1", 10);

  // add a layer, name=sm1, type=Softmax, bottom=fc1, top=sm1
  net.addLayer("sm1", "Softmax", "fc1", "sm1");

  // add a layer, name=cls1, type=NLLLoss, bottom=sm1, top=cls1, label=entryLabelBlob
  net.addLayer("cls1", "ClassNLLLoss", "sm1", "cls1", "entryLabelBlob");

  // add a layer, name=acc1, type=Accuracy, bottom=sm1, top=acc1, label=entryLabelBlob
  net.addLayer("acc1", "ClassAccuracy", "sm1", "acc1", "entryLabelBlob");

Example of network training:

.. code:: cpp

  for (int iteration = 0; iteration < MAX_ITERATION; iteration++) {
    // get batch, input dim = 784, batchsize = 100, (pseudo code)
    get_batch_to("entryDataBlob", 784*100)
    get_batch_to("entryLabelBlob", 100)

    // forward pass of the network (graph)
    net.forward();

    // clear gradient
    net.zeroGrad();

    // backward pass of the network (graph)
    net.backward();

    // report the loss and accuracy
    net.report();

    // parameter update (SGD), learning rate = 1e-3
    net.update(1e-3);
  }

Here is the full example `test_graph_mnist_cls.cc <test_graph_mnist_cls.cc>`__

Documentation
-------------

This is a leight-weight project, please just READ THE CODE.

Dependency and Compilation
--------------------------

This project is designed to use as less library as possible, i.e.
designed from scratch. The only libraries needed by this project are
some auxiliary I/O helper libraries.

* HDF5
* JsonCPP
* OpenMP (if clang++)
* OpenBLAS (Optional)

License
-------

The MIT License.

TODO
----

Want:

* batch norm layer
* rowmajor/column manjor mode
* RNN.
* dynamic graph
* fix destructor and memory leak issue of graph

Postponed:

* Performance Optimization / just link some necessary libs. e.g. OpenBLAS
* Python binding by SWIG. Data interface with Numpy.
* dump and load a trained model via json

Not decided:

* CUDA. Not the original intention. We don't need to write everything again
  if we just want to verify the understanding. cuDNN?
* Elegant data loading approach. I thought a working dirty hack for reading
  data is enough. It is the core part that should keep simple and elegant
  instead of the auxiliary part.
* Model saving/loading in binary mode. ASCII/Json just works everywhere.
* Decaying learning rate. Since the update functions are exposed to the
  user, and the learning rate is an argument of the update function,
  the user may control the learning rate by him/herself.
* Automatic differentiation. Too complex?

Extra Reference
---------------

* Deep Learning Book, Ian Goodfellow, et al.
* Caffe https://github.com/bvlc/caffe
* (Lua)Torch https://github.com/torch/torch7
* PyTorch https://pytorch.org
* Memory leak issue: https://stackoverflow.com/questions/6261201/how-to-find-memory-leak-in-a-c-code-project
* OpenMP issue: https://stackoverflow.com/questions/22634121/openmp-c-matrix-multiplication-run-slower-in-parallel

Changelog
---------

* Nov 8 2017, Draft design of the framework.
* Nov 11 2017, First working version. MLP on MNIST works.
* Nov 17 2017, Convolution/Lenet works. Slowdown the dev pace because the original goal is reached.

Afterword
---------

* Don't create different classes for vectors, matrices and higher-dimentional
  data. Unless you are about to apply specific optimization on them, just
  implement these *D arrays in a common class, say Tensor. In this way, one
  implementaion of the element-wise operations will work for all *D tensors.
* When a network is not working, first check the tensor and layer unit tests.
  Then dump the blobs and the parameter staticsics  during forward and
  backward pass to see if there is anything weird. If so, check the source
  code again to make sure you didn't make any typo, and the arguments are
  correct. Check the code with Valgrind to make sure there is no unexpected
  memory reads and writes. Use GDB if you need fine-grained tracking of what
  the code is doing or backtrace. If the network is still not working, try
  to overfit the model with one or several batches, tune the hyper parameters
  (especially the learning rate), and check the gradient. Besides, one can
  check the gradient with numerical difference method.
* Pay extra attention on the Cache Miss problem when writing code.
* The Netlib BLAS/LAPACK reference implementation is interesting and inspiring.
