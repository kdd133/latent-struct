/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Ublas.h"
#include "Utility.h"
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
  
  SparseRealMat& exponentiate(const AccumLogMat& src, SparseRealMat& dest) {
    assert(dest.size1() == src.size1());
    assert(dest.size2() == src.size2());
    dest.clear();
    AccumLogMat::const_iterator1 it1;
    AccumLogMat::const_iterator2 it2;
    // We use push_back here to efficiently populate the matrix, row by row.
    // See http://www.guwi17.de/ublas/matrix_sparse_usage.html for a rationale.
    for (it1 = src.begin1(); it1 != src.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end(); ++it2)
        dest.push_back(it2.index1(), it2.index2(), exp(*it2));
    return dest;
  }
  
  AccumRealMat& exponentiate(const AccumLogMat& src, AccumRealMat& dest) {
    assert(dest.size1() == src.size1());
    assert(dest.size2() == src.size2());
    dest.clear();
    AccumLogMat::const_iterator1 it1;
    AccumLogMat::const_iterator2 it2;
    for (it1 = src.begin1(); it1 != src.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end(); ++it2)
        dest(it2.index1(), it2.index2()) = exp(*it2);
    return dest;
  }
  
  void addExponentiated(const SparseLogVec& src, SparseRealVec& dest) {
    SparseLogVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) += exp(*it);
  }
  
  void addExponentiated(const SparseLogVec& src, RealVec& dest, double scale) {
    SparseLogVec::const_iterator it;
    for (it = src.begin(); it != src.end(); ++it)
      dest(it.index()) += exp(*it) * scale;
  }
  
  RealVec& subtractWeightVectors(const WeightVector& w, const WeightVector& v,
      RealVec& dest) {
    assert(w.getDim() == v.getDim() && w.getDim() == dest.size());
    for (int i = 0; i < w.getDim(); ++i)
      dest(i) = w.getWeight(i) - v.getWeight(i);
    return dest;
  }
  
  void addOuterProductLowerTriangular(const SparseLogVec& v1,
      const SparseLogVec& v2, LogWeight scale, AccumLogMat& dest) {
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
  
  void addScaled(const AccumRealMat& src, AccumRealMat& dest, double scale) {
    assert(dest.size1() == src.size1());
    assert(dest.size2() == src.size2());
    AccumRealMat::const_iterator1 it1;
    AccumRealMat::const_iterator2 it2;
    for (it1 = src.begin1(); it1 != src.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end(); ++it2)
        dest(it2.index1(), it2.index2()) += (*it2) * scale;
  }
  
  void matrixVectorMultLowerSymmetric(const AccumRealMat& L, const RealVec& x,
      SparseRealVec& b, bool clear_b) {
    if (clear_b)
      b.clear();
    AccumRealMat::const_iterator1 it1;
    AccumRealMat::const_iterator2 it2;
    for (it1 = L.begin1(); it1 != L.end1(); ++it1) {
      for (it2 = it1.begin(); it2 != it1.end(); ++it2) {
        const int i = it2.index1();
        const int j = it2.index2();
        assert(j <= i); // If this fails, L is not lower triangular.
        if (x[j] != 0)
          b(i) += (*it2) * x[j]; // b_i += A_ij * x_i
        if (i != j && x[i] != 0)
          b(j) += (*it2) * x[i]; // b_j += A_ij * x_j
      }
    }
  }
  
  void computeLowerCovarianceMatrix(const AccumLogMat& L, const SparseRealVec& v,
      AccumRealMat& C) {
    C.clear();
    // Note: We iterate over non-zero entries in the matrix v*v' instead of
    // iterating over the entries in L because it is possible to have L(i,j)=0
    // while (v*v')(i,j)!=0. This happens, for example, when a pair of features
    // never co-occur in an alignment, but do occur separately in distinct
    // alignments.
    SparseRealVec::const_iterator it1;
    SparseRealVec::const_iterator it2;
    for (it1 = v.begin(); it1 != v.end(); ++it1) {
      const int i = it1.index();
      for (it2 = v.begin(); it2 != v.end(); ++it2) {
        const int j = it2.index();
        if (j > i)
          continue;
        C(i, j) = exp(L(i, j)) - (v[i] * v[j]);
      }
    }
  }
  
  void scaleMatrixRowsByVecTimesOneMinusVec(AccumRealMat& M, const RealVec& x) {
    AccumRealMat::iterator1 it1;
    AccumRealMat::iterator2 it2;
    for (it1 = M.begin1(); it1 != M.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end(); ++it2) {
        const int i = it2.index1();
        *it2 *= x[i] * (1 - x[i]);
    }
  }
  
  void lowerToSymmetric(AccumLogMat& L) {
    const size_t d = L.size1();
    assert(d == L.size2());
    for (size_t i = 0; i < d; ++i)
      for (size_t j = 0; j < i; ++j)
        L(j, i) = L(i, j);
  }
  
  void sigmoid(const SparseRealVec& x, RealVec& sigma_x) {
    assert(x.size() == sigma_x.size());
    const size_t d = x.size();
    for (size_t i = 0; i < d; ++i)
      if (x[i] != 0)
        sigma_x[i] = Utility::sigmoid(x[i]);
      else
        sigma_x[i] = 0.5;
  }
}
