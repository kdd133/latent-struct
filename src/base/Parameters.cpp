/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Parameters.h"
#include <assert.h>

using std::size_t;

void Parameters::add(const int index, const double v) {
  if (hasU() && index >= w.getDim()) {
    assert(index - w.getDim() >= 0);
    u.add(index - w.getDim(), v);
  }
  else {
    assert(index >= 0);
    w.add(index, v);
  }
}

size_t Parameters::getTotalDim() const {
  return w.getDim() + u.getDim();
}

bool Parameters::hasU() const {
  return u.getDim() > 0;
}

double Parameters::getWeight(int index) const {
  if (hasU() && index >= w.getDim()) {
    assert(index - w.getDim() >= 0);
    return u.getWeight(index - w.getDim());
  }
  else {
    assert(index >= 0);
    return w.getWeight(index);
  }
}

double Parameters::innerProd(const RealVec& fv) const {
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

void Parameters::setParams(const Parameters& other) {
  assert(other.w.getDim() == w.getDim());
  assert(other.u.getDim() == u.getDim());
  w.setWeights(other.w.getWeights(), other.w.getDim());
  if (other.hasU())
    u.setWeights(other.u.getWeights(), other.u.getDim());
}

void Parameters::setWeights(const double* values, int len) {
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

double Parameters::squaredL2Norm() const {
  return w.squaredL2Norm() + u.squaredL2Norm();
}

void Parameters::zero() {
  w.zero();
  u.zero();
}
