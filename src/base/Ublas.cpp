/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Ublas.h"
#include <assert.h>
#include <cmath>

namespace ublas_util {

  SparseLogVec& convertVec(const SparseRealVec& src, SparseLogVec& dest) {
    assert(src.size() == dest.size());
    dest.clear();
    SparseRealVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) = LogWeight(*it);
    return dest;
  }
  
  LogVec& convertVec(const RealVec& src, LogVec& dest) {
    assert(src.size() == dest.size());
    for (size_t i = 0; i < src.size(); ++i)
      dest(i) = LogWeight(src(i));
    return dest;
  }
  
  SparseRealVec& convertVec(const SparseLogVec& src, SparseRealVec& dest) {
    assert(src.size() == dest.size());
    dest.clear();
    SparseLogVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) = exp((double)(*it));
    return dest;
  }
  
  RealVec& convertVec(const LogVec& src, RealVec& dest) {
    assert(src.size() == dest.size());
    for (size_t i = 0; i < src.size(); ++i)
      dest(i) = exp((double)src(i));
    return dest;
  }
}
