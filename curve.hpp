/* curve.hpp for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#if !defined(_LEICHT_CURVE_HPP)
#include "leicht.hpp"
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>

class Curve {
public:
	// list of iter-value tuples
	std::vector<std::pair<size_t, float>> data;

	// Construct a curve object
	Curve(void) { }

	// append a new value to the curve
	void append(size_t iteration, float value) {
		data.push_back(std::pair<size_t, float>(iteration, value));
	}

	// dump to screen or a file
	void dump(std::string fname = "") {
		if (fname == "")
			for (size_t i = 0; i < data.size(); i++)
				std::cout << data[i].first << " " << data[i].second << std::endl;
		else {
			std::ofstream f (fname); assert(f);
			for (size_t i = 0; i < data.size(); i++)
				f << data[i].first << " " << data[i].second << "\n";
			f.close();
		}
	}

	// draw the curve info a file, the format is controled by the file name
	// extension. The extension is parsed by pylab. If you need to customize
	// the generated picture, just modify the generated python file.
	// XXX: Dirty hack, but it works very well.
	void draw(std::string fname) {
		this->dump(fname + ".data");
		std::ofstream py (fname + ".py"); assert(py);
		py << "import pylab" << std::endl;
		py << "curve = pylab.loadtxt('" << fname << ".data')" << std::endl;
		py << "pylab.plot(curve[:,0], curve[:,1])" << std::endl;
		py << "pylab.savefig('" << fname << "')" << endl;
		py.close();
		system(("python3 " + fname + ".py").c_str());
		std::cout << "Curve: curve saved to " << fname << std::endl;
	}

};

#endif // _LEICHT_CURVE_HPP
