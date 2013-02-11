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

class WeightVector;

typedef boost::numeric::ublas::compressed_vector<LogWeight> SparseLogVec;
typedef boost::numeric::ublas::vector<LogWeight> LogVec;
typedef boost::numeric::ublas::compressed_vector<double> SparseRealVec;
typedef boost::numeric::ublas::vector<double> RealVec;

typedef boost::numeric::ublas::compressed_matrix<LogWeight> SparseLogMat;
typedef boost::numeric::ublas::matrix<LogWeight> LogMat;
typedef boost::numeric::ublas::compressed_matrix<double> SparseRealMat;
typedef boost::numeric::ublas::matrix<double> RealMat;

namespace ublas_util {

  SparseLogVec& convertVec(const SparseRealVec& src, SparseLogVec& dest);
  
  LogVec& convertVec(const RealVec& src, LogVec& dest);
  
  SparseRealVec& convertVec(const SparseLogVec& src, SparseRealVec& dest);
  
  SparseRealVec& convertVec(const LogVec& src, SparseRealVec& dest);
  
  RealVec& convertVec(const LogVec& src, RealVec& dest);
  
  SparseRealMat& exponentiate(const SparseLogMat& src, SparseRealMat& dest);
  
  RealVec& subtractWeightVectors(const WeightVector& w, const WeightVector& v,
      RealVec& dest);
  
  // Perform dest += lower(scale*(v1*v2')), where lower(M) returns the lower
  // triangular portion of M.
  void addOuterProductLowerTriangular(const SparseLogVec& v1,
      const SparseLogVec& v2, LogWeight scale, SparseLogMat& dest);
      
  // Perform dest += lower(src), where lower(M) returns the lower triangular
  // portion of M.
  void addLowerTriangular(const SparseLogMat& src, SparseLogMat& dest);
}

#endif
