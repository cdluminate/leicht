/* leicht.hpp -- Main Header for LEICHT --
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#if !defined(_LEICHT_HPP)
#define _LEICHT_HPP

#include <string>
#include <iostream>
#include <sys/time.h>

// [ Version Info ]
#define LEICHT_VERSION "0.1~alpha5"
void leicht_version(void) {
	std::cout << "\x1b[36;1m>>> Using LEICHT " << LEICHT_VERSION << " <<<\x1b[m" << std::endl;
}

// [[ Core 1 : Tensor ]]
// Fundamental data container, i.e. of n-D arrays.
// Including some basic BLAS routines.
#include "tensor.hpp"

// [[ Core 2 : Blob ]]
// Combination of two blobs, one for value and another for gradient.
// Used as Nodes in the network graph.
#include "blob.hpp"

// [[ Core 3 : Layer ]]
// Network layers that operates on the blobs.
// Used as Edges in the network graph.
#include "layer.hpp"

// [[ Core 4: Graph ]]
// Network graph, where the nodes are blobs, edges are layers.
// The graph is directed acyclic graph (DAG) (when unfolded).
// The graph is static graph.
#include "graph.hpp"
//#include "dygraph.hpp" // FIXME

// [[ Auxiliary 1: Dataloader, i.e I/O Helper ]]
#include "dataloader.hpp"

// [[ Auxiliary 2: Curve Drawing Helper ]]
#include "curve.hpp"

// helper: highlight banner
inline void leicht_bar_train(long iter) {
	std::cerr << "\x1b[33;1m>>>\x1b[m Training @ Iteration \x1b[33;1m" << iter << " ::\x1b[m" << endl;
}
inline void leicht_bar_val(long iter) {
	std::cerr << "\x1b[35;1m>>>\x1b[m Validation @ Iteration \x1b[35;1m" << iter << " ::\x1b[m" << endl;
}

// helper
inline void leicht_threads(size_t n) {
#if defined(USE_OPENMP)
	omp_set_num_threads(n); // or setup with OMP_NUM_THREADS
#endif
}

// helper: Timer
// XXX: don't do too much printint between tic and toc.
static struct timeval _leicht_helper_tv;
void tic(void) {
	// save current time
	fprintf(stderr, "%s:%d] Timer: Stopwatch Started.\n",
			__FILE__, __LINE__);

	gettimeofday(&_leicht_helper_tv, nullptr);
}
void toc(void) { // FIXME: return a value?
	// get current time
	struct timeval now;
	gettimeofday(&now, nullptr);
	// report time elapsed
	fprintf(stderr, "%s:%d] Timer: Elapsed \x1b[31;1m%.3lf\x1b[m miliSec (1e-3 Sec).\n",
			__FILE__, __LINE__,
			(now.tv_sec*1e6 + now.tv_usec - _leicht_helper_tv.tv_sec*1e6
			 - _leicht_helper_tv.tv_usec)/1e3);
}
#endif // _LEICHT_HPP
