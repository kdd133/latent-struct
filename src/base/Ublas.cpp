/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Ublas.h"
#include "WeightVector.h"
#include <assert.h>
#include <cmath>

namespace ublas_util {

  SparseLogVec& logarithm(const SparseRealVec& src, SparseLogVec& dest) {
    assert(dest.size() >= src.size());
    dest.clear();
    SparseRealVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) = LogWeight(*it);
    return dest;
  }
  
  LogVec& logarithm(const RealVec& src, LogVec& dest) {
    assert(dest.size() >= src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dest(i) = LogWeight(src(i));
    return dest;
  }
  
  SparseRealVec& exponentiate(const SparseLogVec& src, SparseRealVec& dest) {
    assert(dest.size() >= src.size());
    dest.clear();
    SparseLogVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) = exp((double)(*it));
    return dest;
  }
  
  RealVec& exponentiate(const LogVec& src, RealVec& dest) {
    assert(dest.size() >= src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dest(i) = exp((double)src(i));
    return dest;
  }
  
  SparseRealMat& exponentiate(const SparseLogMat& src, SparseRealMat& dest) {
    assert(dest.size1() == src.size1());
    assert(dest.size2() == src.size2());
    dest.clear();
    SparseLogMat::const_iterator1 it1;
    SparseLogMat::const_iterator2 it2;
    // We use push_back here to efficiently populate the matrix, row by row.
    // See http://www.guwi17.de/ublas/matrix_sparse_usage.html for a rationale.
    for (it1 = src.begin1(); it1 != src.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end(); ++it2)
        dest.push_back(it2.index1(), it2.index2(), exp(*it2));
    return dest;
  }
  
  RealVec& subtractWeightVectors(const WeightVector& w, const WeightVector& v,
      RealVec& dest) {
    assert(w.getDim() == v.getDim() && w.getDim() == dest.size());
    for (int i = 0; i < w.getDim(); ++i)
      dest(i) = w.getWeight(i) - v.getWeight(i);
    return dest;
  }
  
  void addOuterProductLowerTriangular(const SparseLogVec& v1,
      const SparseLogVec& v2, LogWeight scale, SparseLogMat& dest) {
    assert(v1.size() == v2.size());
    assert(dest.size1() == v1.size());
    assert(dest.size2() == v1.size());
    SparseLogVec::const_iterator it1;
    SparseLogVec::const_iterator it2;
    for (it1 = v1.begin(); it1 != v1.end(); ++it1)
      for (it2 = v2.begin(); it2 != v2.end() && it2.index() <= it1.index();
          ++it2) {
        dest(it1.index(), it2.index()) += (*it1) * (*it2) * scale;
      }
  }
  
  void addLowerTriangular(const SparseLogMat& src, SparseLogMat& dest) {
    assert(dest.size1() == src.size1());
    assert(dest.size2() == src.size2());
    SparseLogMat::const_iterator1 it1;
    SparseLogMat::const_iterator2 it2;
    for (it1 = src.begin1(); it1 != src.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end() && it2.index2() <= it2.index1();
          ++it2) {
        dest(it2.index1(), it2.index2()) += *it2;
      }
  }
}
