// http://swig.org/Doc3.0/SWIGDocumentation.html
%module leicht

%{

#include "leicht.hpp"

%}

%include "leicht.hpp"
%include "tensor.hpp"
%include "blob.hpp"
%include "layer.hpp"

namespace llas {
    %template(dasum) asum<double>;
    %template(sasum) asum<float>;
}

%template(fp64Tensor) Tensor<double>;
%template(fp32Tensor) Tensor<float>;

%template(fp64Blob) Blob<double>;
%template(fp32Blob) Blob<float>;

%template(fp64Layer) Layer<double>;
%template(fp32Layer) Layer<float>;

 %template(fp64LinearLayer)         LinearLayer<double>;
 %template(fp64Conv2dLayer)         Conv2dLayer<double>;
 %template(fp64ReluLayer)           ReluLayer<double>;
 %template(fp64SoftmaxLayer)        SoftmaxLayer<double>;
 %template(fp64MSELoss)             MSELoss<double>;
 %template(fp64ClassNLLLoss)        ClassNLLLoss<double>;
 %template(fp64ClassAccuracy)       ClassAccuracy<double>;
 %template(fp64MaxpoolLayer)        MaxpoolLayer<double>;
 %template(fp64TransposeLayer)      TransposeLayer<double>;
