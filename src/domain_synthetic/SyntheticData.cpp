/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Dataset.h"
#include "Example.h"
#include "SparsePattern.h"
#include "SyntheticData.h"
#include "Ublas.h"
#include <assert.h>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/shared_array.hpp>
#include <cmath>
#include <limits>

using namespace boost;
using namespace std;
using numeric::ublas::diagonal_matrix;
using numeric::ublas::scalar_vector;

namespace synthetic {

void generate(size_t t, size_t nx, size_t ny, size_t nz,
    const Parameters& theta, Dataset& dataset, int seed) {  
  // Create two random number generators: one for a normal distribution and
  // another for a uniform distribution over [0,1).
  mt19937 mersenne(seed);
  normal_distribution<> normal;
  variate_generator<mt19937, normal_distribution<> > randn(mersenne, normal);  
  uniform_01<> uniform;
  variate_generator<mt19937, uniform_01<> > rand(mersenne, uniform); 

  for (size_t i = 0; i < t; ++i) {  
    // Generate a random observation.
    SparseRealVec x(nx);
    for (size_t k = 0; k < nx; ++k)
      x(k) = randn();
    
    SparseRealVec probs_yz = prob_x(theta, x, ny, nz); // p(y,z|x)
    SparseRealVec cummprobs_yz = cumsum(probs_yz);
    const int index = first_index_gt(cummprobs_yz, rand()); // sample y,z|x
    assert(index >= 0);

    // Convert the sampled index into its corresponding y and z labels. 
    int y = -1, z = -1;
    ind2sub(ny, nz, index, y, z);
    
    // Create an example and add it to the dataset.
    Example xi(new SparsePattern(x, z), y);
    dataset.addExample(xi);
  }
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

SparseRealVec prob_xy(const Parameters& theta, const SparseRealVec& x,
    size_t y, size_t ny, size_t nz) {
  SparseRealVec probs(nz);
  for (size_t z = 0; z < nz; ++z)
    probs(z) = theta.innerProd(phi_rep(x, y, z, ny, nz));
  const double A = log_sum_exp(probs);
  probs -= scalar_vector<double>(probs.size(), A);
  vec_exp(probs);
  assert(abs(vec_sum(probs)) - 1 < 1e-8); // verify that entries sum to 1
  return probs;
}

SparseRealVec phi_mean_x(const Parameters& theta, const SparseRealVec& x,
    size_t ny, size_t nz) {
  const size_t n = theta.w.getDim();
  const size_t nyz = ny*nz;
  
  SparseRealMat Phi_yz(nyz, n);
  for (size_t y = 0; y < ny; ++y)
    for (size_t z = 0; z < nz; ++z)
      row(Phi_yz, y*nz + z) = phi_rep(x, y, z, ny, nz);
      
  const SparseRealVec probs_yz = prob_x(theta, x, ny, nz);  
  SparseRealVec phi_mean(n);
  axpy_prod(probs_yz, Phi_yz, phi_mean);
  return phi_mean;
}

SparseRealVec phi_mean_xy(const Parameters& theta, const SparseRealVec& x,
    std::size_t y, std::size_t ny, std::size_t nz) {
  const size_t n = theta.w.getDim();
  // TODO: Implement this function.
  return SparseRealVec(n);
}

SparseRealMat phi_Cov_x(const Parameters& theta, const SparseRealVec& x,
    size_t ny, size_t nz) {
  const size_t n = theta.w.getDim();
  const size_t nyz = ny*nz;
  
  SparseRealMat Phi_yz(nyz, n);  
  for (size_t y = 0; y < ny; ++y)
    for (size_t z = 0; z < nz; ++z)
      row(Phi_yz, y*nz + z) = phi_rep(x, y, z, ny, nz);
  
  const SparseRealVec probs_yz = prob_x(theta, x, ny, nz);
  
  // Compute probs_yz*Phi_yz and store the result in phi_mean.
  SparseRealVec phi_mean(n);
  axpy_prod(probs_yz, Phi_yz, phi_mean);
  
  // Create a diagonal matrix from the entries of the vector probs_yz.
  diagonal_matrix<double> diag_probs_yz(nyz);
  for (size_t i = 0; i < nyz; ++i)
    diag_probs_yz(i, i) = probs_yz(i);

  // Compute Phi_yz'*diag(probs_yz)*Phi_yz and store the result in phi2_mean.
  SparseRealMat left(n, nyz);
  axpy_prod(trans(Phi_yz), diag_probs_yz, left);
  SparseRealMat phi2_mean(n, n);
  axpy_prod(left, Phi_yz, phi2_mean);
  
  return phi2_mean - outer_prod(phi_mean, phi_mean);
}

SparseRealMat phi_Cov_xy(const Parameters& theta, const SparseRealVec& x,
    size_t y, size_t ny, size_t nz) {
  const size_t n = theta.w.getDim();
  
  SparseRealMat Phi_z(nz, n);
  for (size_t z = 0; z < nz; ++z)
    row(Phi_z, z) = phi_rep(x, y, z, ny, nz);
    
  const SparseRealVec probs_z = prob_xy(theta, x, y, ny, nz);
  
  // Compute probs_z*Phi_z and store the result in phi_mean.
  SparseRealVec phi_mean(n);
  axpy_prod(probs_z, Phi_z, phi_mean);
  
  // Create a diagonal matrix from the entries of the vector probs_z.
  diagonal_matrix<double> diag_probs_z(nz);
  for (size_t i = 0; i < nz; ++i)
    diag_probs_z(i, i) = probs_z(i);
    
  // Compute Phi_z'*diag(probs_z)*Phi_z and store the result in phi2_mean.
  SparseRealMat left(n, nz);
  axpy_prod(trans(Phi_z), diag_probs_z, left);
  SparseRealMat phi2_mean(n, n);
  axpy_prod(left, Phi_z, phi2_mean);
  
  return phi2_mean - outer_prod(phi_mean, phi_mean);
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

void ind2sub(int d1, int d2, int index, int& i, int& j) {
  assert(d1 > 0 && d2 > 0 && index >= 0);
  i = index / d2;
  j = index % d2;
}

}
