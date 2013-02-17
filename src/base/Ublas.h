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
#include <boost/numeric/ublas/vector_of_vector.hpp>
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

// A matrix type that is more efficient for += addition; i.e., for accumulating
// counts.
typedef boost::numeric::ublas::generalized_vector_of_vector<double,
    boost::numeric::ublas::row_major, boost::numeric::ublas::vector<
    boost::numeric::ublas::compressed_vector<double> > > AccumLogMat;

namespace ublas_util {

  SparseLogVec& logarithm(const SparseRealVec& src, SparseLogVec& dest);
  
  LogVec& logarithm(const RealVec& src, LogVec& dest);
  
  SparseRealVec& exponentiate(const SparseLogVec& src, SparseRealVec& dest);
  
  RealVec& exponentiate(const LogVec& src, RealVec& dest);
  
  SparseRealMat& exponentiate(const SparseLogMat& src, SparseRealMat& dest);
  
  SparseRealMat& exponentiate(const AccumLogMat& src, SparseRealMat& dest);
  
  RealVec& subtractWeightVectors(const WeightVector& w, const WeightVector& v,
      RealVec& dest);
  
  // Perform dest += lower(scale*(v1*v2')), where lower(M) returns the lower
  // triangular portion of M.
  void addOuterProductLowerTriangular(const SparseLogVec& v1,
      const SparseLogVec& v2, LogWeight scale, AccumLogMat& dest);
      
  // Perform dest += lower(src), where lower(M) returns the lower triangular
  // portion of M.
  void addLowerTriangular(const SparseLogMat& src, SparseLogMat& dest);
  
  void setEntriesToZero(SparseLogMat& M);
  
  void setEntriesToZero(AccumLogMat& M);
}

#endif
