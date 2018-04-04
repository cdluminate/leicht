/* Dataloader.cc for LEICHT
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
#if !defined(_LEICHT_DATALOADER_H)
#define _LEICHT_DATALOADER_H

#include <string>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdio>

#include "H5Cpp.h"

using namespace std;
using namespace H5;

// read the whole 2D dataset into memory
template <typename Dtype>
void
leicht_hdf5_read(
		H5std_string name_h5file,
		H5std_string name_dataset,
		size_t offset1, size_t offset2,
		size_t count1,  size_t count2,
		Dtype* dest)
{
	H5File h5file (name_h5file, H5F_ACC_RDONLY);
	DataSet dataset = h5file.openDataSet(name_dataset);
//	H5T_class_t type_class = dataset.getTypeClass();
	DataSpace dataspace = dataset.getSpace();

	hsize_t offset[2] = {offset1, offset2};
	hsize_t count[2] = {count1, count2};
	dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

	hsize_t dimsm[2] = {count1, count2};
	DataSpace memspace(2, dimsm);

	hsize_t offset_out[2] = {offset1, offset2};
	hsize_t count_out[2] = {count1, count2};
	memspace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);

	dataset.read(dest, PredType::NATIVE_DOUBLE, memspace, dataspace);
}

// read the whole 1D dataset into memory
template <typename Dtype>
void
leicht_hdf5_read(
		H5std_string name_h5file,
		H5std_string name_dataset,
		size_t offset1,
		size_t count1,
		Dtype* dest)
{
	H5File h5file (name_h5file, H5F_ACC_RDONLY);
	DataSet dataset = h5file.openDataSet(name_dataset);
//	H5T_class_t type_class = dataset.getTypeClass();
	DataSpace dataspace = dataset.getSpace();

	hsize_t offset[1] = {offset1};
	hsize_t count[1] = {count1};
	dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

	hsize_t dimsm[1] = {count1,};
	DataSpace memspace(1, dimsm);

	hsize_t offset_out[1] = {offset1,};
	hsize_t count_out[1] = {count1,};
	memspace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);

	dataset.read(dest, PredType::NATIVE_DOUBLE, memspace, dataspace);
}
#endif // defined(_LEICHT_DATALOADER_H)

#if defined(LEICHT_TEST_DATALOADER)
int
main()
{
	double data_out[10][784];
	memset(data_out, 0x0, 10*784*sizeof(double));
	leicht_hdf5_read("demo.h5", "data", 0, 0, 10, 784, data_out);
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 784; j++) {
			printf(" %7.4f", data_out[i][j]);
		}
		cout << endl;
	}
	double data_out2[10];
	memset(data_out2, 0x0, 10*sizeof(double));
	leicht_hdf5_read("demo.h5", "label", 0, 10, data_out2);
	for (int j = 0; j < 10; j++) {
		printf(" %7.4f", data_out2[j]);
	}

	return 0;
}
#endif
