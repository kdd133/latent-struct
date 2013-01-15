/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "SparsePattern.h"
#include "SyntheticData.h"
#include "Ublas.h"
#include "Utility.h"
#include <boost/shared_array.hpp>

using namespace boost;
using namespace std;

void SyntheticData::generate(size_t m, size_t nx, size_t ny, size_t nz,
    const Parameters& theta, Dataset& dataset, int seed) {

  SparsePattern x(Utility::generateGaussianSamples(nx, 0, 1, seed), nx);
  
}

SparseRealVec SyntheticData::prob_x(const Parameters& theta,
    const SparsePattern& x, size_t ny, size_t nz) {
  SparseRealVec response // wait, isn't this supposed to be a matrix?
}
