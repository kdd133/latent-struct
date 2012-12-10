/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _UBLAS_H
#define _UBLAS_H

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "LogWeight.h"
#include <assert.h>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>

typedef boost::numeric::ublas::compressed_vector<LogWeight> SparseLogVec;
typedef boost::numeric::ublas::vector<LogWeight> LogVec;
typedef boost::numeric::ublas::compressed_vector<double> SparseRealVec;
typedef boost::numeric::ublas::vector<double> RealVec;

typedef boost::numeric::ublas::compressed_matrix<LogWeight> SparseLogMat;
typedef boost::numeric::ublas::matrix<LogWeight> LogMat;

namespace ublas_util {

  SparseLogVec& convertVec(const SparseRealVec& src, SparseLogVec& dest);
  
  LogVec& convertVec(const RealVec& src, LogVec& dest);
  
  SparseRealVec& convertVec(const SparseLogVec& src, SparseRealVec& dest);
  
  RealVec& convertVec(const LogVec& src, RealVec& dest);
}

#endif