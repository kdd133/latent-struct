/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _SYNTHETIC_DATA_H
#define _SYNTHETIC_DATA_H

#include "Dataset.h"
#include "Parameters.h"
#include "Ublas.h"

namespace synthetic {

void generate(std::size_t t, std::size_t nx, std::size_t ny, std::size_t nz,
    const Parameters& theta, Dataset& dataset, int seed = 0);
      
SparseRealVec phi_rep(const SparseRealVec& x, std::size_t y, std::size_t z,
    std::size_t ny, std::size_t nz);
      
SparseRealVec prob_x(const Parameters& theta, const SparseRealVec& x,
    std::size_t ny, std::size_t nz);

SparseRealVec prob_xy(const Parameters& theta, const SparseRealVec& x,
    std::size_t y, std::size_t ny, std::size_t nz);

SparseRealVec phi_mean_x(const Parameters& theta, const SparseRealVec& x,
    std::size_t ny, std::size_t nz);
    
SparseRealVec phi_mean_xy(const Parameters& theta, const SparseRealVec& x,
    std::size_t y, std::size_t ny, std::size_t nz);

SparseRealMat phi_Cov_x(const Parameters& theta, const SparseRealVec& x,
    std::size_t ny, std::size_t nz);
    
SparseRealMat phi_Cov_xy(const Parameters& theta, const SparseRealVec& x,
    std::size_t y, std::size_t ny, std::size_t nz);

double log_sum_exp(const SparseRealVec& x);

SparseRealVec& vec_exp(SparseRealVec& x);

SparseRealVec& vec_log(SparseRealVec& x);

double vec_sum(const SparseRealVec& x);

SparseRealVec cumsum(const SparseRealVec& x);

int first_index_gt(const SparseRealVec& x, const double value);

void ind2sub(int dim1, int dim2, int index, int& i, int& j);

}

#endif
