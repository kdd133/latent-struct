/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _PARAMETERS_H
#define _PARAMETERS_H

#include "Ublas.h"
#include "WeightVector.h"
#include <string>

// Parameters is a data structure that stores WeightVector objects w and u,
// which are understood to be concatenated as [w u]. Note that the u portion of
// the vector may be empty, as some training objectives use only w.

class Parameters {

public:
  WeightVector w;
  WeightVector u;
  
  Parameters() { }
  Parameters(int dw) : w(dw) { }
  Parameters(int dw, int du) : w(dw), u(du) { }
  
  void add(const int index, const double v) {
    if (hasU() && index >= w.getDim()) {
      assert(index - w.getDim() >= 0);
      u.add(index - w.getDim(), v);
    }
    else {
      assert(index >= 0);
      w.add(index, v);
    }
  }
  
  // Return the "total dimensionality" of the parameters; i.e., the sum of the
  // dimensionalities of the component vectors.
  std::size_t getTotalDim() const {
    return w.getDim() + u.getDim();
  }
  
  bool hasU() const {
    return u.getDim() > 0;
  }
  
  double getWeight(int index) const {
    if (hasU() && index >= w.getDim()) {
      assert(index - w.getDim() >= 0);
      return u.getWeight(index - w.getDim());
    }
    else {
      assert(index >= 0);
      return w.getWeight(index);
    }
  }
  
  double innerProd(const RealVec& fv) const {
    const int len = fv.size();
    if (len == w.getDim()) {
      // w only
      return w.innerProd(fv);
    }
    else {
      // w and u
      assert(hasU() && len == w.getDim() + u.getDim());
      double prod = 0;
      for (int i = 0; i < w.getDim(); i++)
        prod += fv(i) * w.getWeight(i);
      int fv_i = w.getDim();
      for (int i = 0; fv_i < len; i++, fv_i++)
        prod += fv(fv_i) * u.getWeight(i);
      return prod;
    }
  }
  
  void setParams(const Parameters& other) {
    assert(other.w.getDim() == w.getDim());
    assert(other.u.getDim() == u.getDim());
    w.setWeights(other.w.getWeights(), other.w.getDim());
    if (other.hasU())
      u.setWeights(other.u.getWeights(), other.u.getDim());
  }
  
  void setWeights(const double* values, int len) {
    if (len == w.getDim()) {
      assert(!hasU());
      // w only
      w.setWeights(values, w.getDim());
    }
    else {
      // w and u
      assert(hasU() && len == w.getDim() + u.getDim());
      w.setWeights(values, w.getDim());
      u.setWeights(values + w.getDim(), u.getDim());
    }
  }
  
  double squaredL2Norm() const {
    return w.squaredL2Norm() + u.squaredL2Norm();
  }
  
  void zero() {
    w.zero();
    u.zero();
  }
};

#endif
