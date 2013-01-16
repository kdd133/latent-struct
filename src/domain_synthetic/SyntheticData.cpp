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
#include <assert.h>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/shared_array.hpp>
#include <cmath>
#include <limits>

using namespace boost;
using namespace std;
using numeric::ublas::scalar_vector;

namespace synthetic {

void generate(size_t m, size_t nx, size_t ny, size_t nz,
    const Parameters& theta, Dataset& dataset, int seed) {
  
  // Create two random number generators: one for a normal distribution and
  // another for a uniform distribution over [0,1).
  mt19937 mersenne(seed);
  normal_distribution<> normal;
  variate_generator<mt19937, normal_distribution<> > randn(mersenne, normal);  
  uniform_01<> uniform;
  variate_generator<mt19937, uniform_01<> > rand(mersenne, uniform); 
  
  // Generate a random vector.
  SparseRealVec x(nx);
  for (SparseRealVec::iterator it = x.begin(); it != x.end(); ++it)
    *it = randn();
  
  SparseRealVec probs_yz = prob_x(theta, x, ny, nz);
  SparseRealVec cummprobs_yz = cumsum(probs_yz);
  const int index = first_index_gt(x, rand());
}

int first_index_gt(const SparseRealVec& x, const double value) {
  for (size_t i = 0; i < x.size(); ++i)
    if (x[i] > value)
      return i;
  return -1;
}

SparseRealVec phi_rep(const SparseRealVec& x, size_t y, size_t z, size_t ny,
    size_t nz) {
  const int nx = x.size();
  const int ind_yz = y*nz + z;
  SparseRealVec phi(nx*ny*nz);
  subrange(phi, nx*ind_yz, nx*(ind_yz+1)) = x;
  return phi;
}

SparseRealVec prob_x(const Parameters& theta, const SparseRealVec& x, size_t ny,
    size_t nz) {
  SparseRealVec probs(nz*ny);
  for (size_t y = 0; y < ny; ++y)
    for (size_t z = 0; z < nz; ++z)
      probs(y*nz + z) = theta.innerProd(phi_rep(x, y, z, ny, nz));
  const double A = log_sum_exp(probs);
  probs -= scalar_vector<double>(probs.size(), A);
  vec_exp(probs);
  assert(abs(vec_sum(probs)) - 1 < 1e-8); // verify that entries sum to 1
  return probs;
}

double log_sum_exp(const SparseRealVec& x) {
  // TODO: Prevent overflow/underflow.
  SparseRealVec copy_x = x;
  return log(vec_sum(vec_exp(copy_x)));
}

SparseRealVec& vec_exp(SparseRealVec& x) {
  SparseRealVec::iterator it;
  for (it = x.begin(); it != x.end(); ++it)
    *it = exp(*it);
  return x;
}

SparseRealVec& vec_log(SparseRealVec& x) {
  SparseRealVec::iterator it;
  for (it = x.begin(); it != x.end(); ++it)
    *it = log(*it);
  return x;
}

double vec_sum(const SparseRealVec& x) {
  double sum = 0;
  SparseRealVec::const_iterator it;
  for (it = x.begin(); it != x.end(); ++it)
    sum += *it;
  return sum;
}

SparseRealVec cumsum(const SparseRealVec& x) {
  SparseRealVec cumm(x.size());
  double sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i];
    cumm(i) = sum;
  }
  return cumm;
}

void ind2sub(int d1, int d2, int ndx, int& i_out, int& j_out) {
  // TODO: Work this out on paper, keeping in mind that Matlab indexes from 1.
}

}
