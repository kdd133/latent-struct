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

  SparseLogVec& convertVec(const SparseRealVec& src, SparseLogVec& dest) {
    assert(dest.size() >= src.size());
    dest.clear();
    SparseRealVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) = LogWeight(*it);
    return dest;
  }
  
  LogVec& convertVec(const RealVec& src, LogVec& dest) {
    assert(dest.size() >= src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dest(i) = LogWeight(src(i));
    return dest;
  }
  
  SparseRealVec& convertVec(const SparseLogVec& src, SparseRealVec& dest) {
    assert(dest.size() >= src.size());
    dest.clear();
    SparseLogVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) = exp((double)(*it));
    return dest;
  }
  
  RealVec& convertVec(const LogVec& src, RealVec& dest) {
    assert(dest.size() >= src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dest(i) = exp((double)src(i));
    return dest;
  }
  
  RealMat& exponentiate(const SparseLogMat& src, RealMat& dest) {
    assert(dest.size1() == src.size1());
    assert(dest.size2() == src.size2());
    dest.clear();
    SparseLogMat::const_iterator1 it1;
    SparseLogMat::const_iterator2 it2;
    for (it1 = src.begin1(); it1 != src.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end(); ++it2)
        dest(it2.index1(), it2.index2()) = exp(*it2);
    return dest;
  }
  
  RealVec& subtractWeightVectors(const WeightVector& w, const WeightVector& v,
      RealVec& dest) {
    assert(w.getDim() == v.getDim() && w.getDim() == dest.size());
    for (int i = 0; i < w.getDim(); ++i)
      dest(i) = w.getWeight(i) - v.getWeight(i);
    return dest;
  }
}
