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
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/shared_array.hpp>

using namespace boost;
using namespace std;

void SyntheticData::generate(size_t m, size_t nx, size_t ny, size_t nz,
    const Parameters& theta, Dataset& dataset, int seed) {

  SparsePattern x(Utility::generateGaussianSamples(nx, 0, 1, seed), nx);
  
}

SparseRealVec SyntheticData::phi_rep(const SparsePattern& x, size_t y, size_t z,
    size_t ny, size_t nz) {
  const int nx = x.getSize();
  const int ind_yz = y*nz + z;
  SparseRealVec phi(nx*ny*nz);
  subrange(phi, nx*ind_yz, nx*(ind_yz+1)) = x.getVector();
  return phi;
}

SparseRealVec SyntheticData::prob_x(const Parameters& theta,
    const SparsePattern& x, size_t ny, size_t nz) {
  SparseRealVec response(nz*ny);
  for (size_t y = 0; y < ny; ++y)
    for (size_t z = 0; z < nz; ++z)
      response(y*nz + z) = theta.innerProd(phi_rep(x, y, z, ny, nz));
  return response;
}
