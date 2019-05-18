CXX ?= clang++
INCLUDES = -I/usr/include/hdf5/serial
LIBS = -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_serial -ljsoncpp -lblas
OPENMP = -fopenmp -DUSE_OPENMP # Comment this out to disable openmp
BLAS = -DUSE_BLAS # Comment this out to disable openblas
CXXFLAGS += $(OPENMP) $(BLAS) -std=c++11 -Wall -g -O2 -march=native #-fno-inline-functions -Og
VG ?= valgrind --leak-check=yes

gtest:
	g++ tests/lucs.cc -I./include -lgtest -lpthread -o lucs-gtest
	./lucs-gtest

.PHONY: test unittest layertest
# -- e.g. make unittest >/dev/null
unittest: ut_tensor ut_blob ut_layer ut_graph
layertest: test_layer_lineq test_layer_mnist_cls test_layer_mnist_reg
graphtest: test_graph_mnist_cls test_graph_mnist_cls2

# -- Fake dataset generator
demo.h5:
	python3 gen-demoh5.py
mnist.fake.h5:
	python3 gen-demoh5.py mnist

# -- Unit Tests
ut_tensor: tensor.hpp demo.h5
	python3 leichtut.py $<
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o $@.elf $<_ut.cc
	$(VG) ./$@.elf
ut_blob: blob.hpp
	python3 leichtut.py $<
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o $@.elf $<_ut.cc
	$(VG) ./$@.elf
ut_layer: layer.hpp
	python3 leichtut.py $<
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o $@.elf $<_ut.cc
	$(VG) ./$@.elf
ut_graph: graph.hpp
	python3 leichtut.py $<
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o $@.elf $<_ut.cc
	$(VG) ./$@.elf

# -- Feel the horror in the performance difference
bench_tensor:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o $@.elf $@.cc -lblas
	./$@.elf

# -- Layer-level Tests
test_layer_lineq:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_layer_lineq.elf test_layer_lineq.cc
	$(VG) ./test_layer_lineq.elf #OK
test_layer_mnist_reg:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_layer_mnist_reg.elf test_layer_mnist_reg.cc
	$(VG) ./test_layer_mnist_reg.elf #OK
test_layer_mnist_cls:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_layer_mnist_cls.elf test_layer_mnist_cls.cc
	$(VG) ./test_layer_mnist_cls.elf #OK
test_layer_lenet_cls:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_layer_lenet_cls.elf test_layer_lenet_cls.cc
	$(VG) ./test_layer_lenet_cls.elf #FIXME

# -- Graph-level Tests
test_graph_mnist_cls:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o test_graph_mnist_cls.elf test_graph_mnist_cls.cc
	$(VG) ./test_graph_mnist_cls.elf # working, but FIXME: Memory issue
test_graph_mnist_reg:
	$(CXX) $(INCLUDES) $(LIBS) $(CXXFLAGS) -o $@.elf $@.cc
	$(VG) ./$@.elf # working, but FIXME: Memory issue

# -- Python Binding
swig:
	CFLAGS="$(CXXFLAGS)" python3 setup.py build
	-cp build/lib.linux-x86_64-3.6/leicht.py leicht.py
	cp build/lib.linux-x86_64-3.6/leicht.*.so _leicht.so
pyunit: swig
	python3 test_pyunit_tensor.py -v

# -- Benchmarks
.PHONY: benchmark
BASEFLAG= -std=c++11 -Wall -fopenmp -DUSE_OPENMP
benchmark: mnist.fake.h5
	# change the code to use the fake dataset, and change iterations
	cp test_mnist_cls.cc test_benchmark.cc
	sed -i -e 's/mnist.h5/mnist.fake/g' test_benchmark.cc
	@echo
	# you can use "time" instead of "perf"
	
	# warm up and report nothing, the kernel may cache something
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -o benchmark.elf test_benchmark.cc
	./benchmark.elf > /dev/null
	@echo

	# compile with -O0 and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O0 -march=native and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -march=native -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O0 -march=native -flto and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O0 -march=native -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 -march=native and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -march=native -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 -flto and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O2 -march=native -flto and test
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O2 -march=native -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

	# compile with -O3 -march=native -flto and test, some times the aggressive optimization may not improve performance.
	$(CXX) $(INCLUDES) $(LIBS) $(BASEFLAG) -O3 -march=native -flto -o benchmark.elf test_benchmark.cc
	sudo perf stat ./benchmark.elf > /dev/null

clean:
	-$(RM) demo.h5 *.elf test.leicht *_ut.cc
	-$(RM) leicht.py _leicht.so leicht_wrap.cpp
	-$(RM) -rf __pycache__/ build
