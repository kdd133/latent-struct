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

void Parameters::init() {
  assert(_numWV == 4);
  _wv[0] = &w;
  _wv[1] = &u;
  _wv[2] = &shared_w;
  _wv[3] = &shared_u;
}

void Parameters::add(const int index, const double v) {
  assert(index >= 0 && index < getDimTotal());
  int offset = 0;
  for (int i = 0; i < _numWV; i++) {
    if (index < offset + _wv[i]->getDim()) {
      assert(index - offset >= 0);
      _wv[i]->add(index - offset, v);
      break;
    }
    else
      offset += _wv[i]->getDim();
  }
}

size_t Parameters::getDimWU() const {
  return w.getDim() + u.getDim();
}

size_t Parameters::getDimTotal() const {
  return w.getDim() + u.getDim() + shared_w.getDim() + shared_u.getDim();
}

bool Parameters::hasU() const {
  return u.getDim() > 0;
}

bool Parameters::hasSharedW() const {
  return shared_w.getDim() > 0;
}

bool Parameters::hasSharedU() const {
  return shared_u.getDim() > 0;
}

const double& Parameters::operator[](int index) const {
  assert(index >= 0 && index < getDimTotal());
  int offset = 0;
  for (int i = 0; i < _numWV; i++) {
    if (index < offset + _wv[i]->getDim()) {
      assert(index - offset >= 0);
      return (*_wv[i])[index - offset];
    }
    else
      offset += _wv[i]->getDim();
  }
  assert(0); // should never reach this point
  return w[0];
}

double Parameters::innerProd(const RealVec& fv) const {
  double prod = 0;
  int fv_j = 0;
  for (int i = 0; i < _numWV; i++) {
    if (_wv[i]->getDim() > 0) {
      if (fv_j + _wv[i]->getDim() > fv.size())
        break;
      for (int j = 0; j < _wv[i]->getDim(); j++, fv_j++)
        prod += fv(fv_j) * (*_wv[i])[j];
    }
  }
  return prod;
}

void Parameters::setParams(const Parameters& other) {
  assert(w.getDim() > 0);
  w.setWeights(other.w.getWeights(), other.w.getDim());
  if (other.hasU())
    u.setWeights(other.u.getWeights(), other.u.getDim());
  if (other.hasSharedW())
    shared_w.setWeights(other.shared_w.getWeights(), other.shared_w.getDim());
  if (other.hasSharedU())
    shared_u.setWeights(other.shared_u.getWeights(), other.shared_u.getDim());
  init();
}

void Parameters::setWeights(const double* values, int len) {
  assert(values && len > 0 && len <= getDimTotal());
  int offset = 0;
  for (int i = 0; i < _numWV && offset + _wv[i]->getDim() <= len; i++) {
    const double* first = values + offset; 
    _wv[i]->setWeights(first, _wv[i]->getDim());
    offset += _wv[i]->getDim();
  }
}

double Parameters::squaredL2Norm() const {
  return w.squaredL2Norm() + u.squaredL2Norm() + shared_w.squaredL2Norm() +
      shared_u.squaredL2Norm();
}

void Parameters::zero() {
  w.zero();
  u.zero();
  shared_w.zero();
  shared_u.zero();
}

int Parameters::indexW() const {
  return 0;
}

int Parameters::indexU() const {
  if (!hasU())
    return -1;
  return w.getDim();
}

int Parameters::indexSharedW() const {
  if (!hasSharedW())
    return -1;
  return indexU() + u.getDim();
}

int Parameters::indexSharedU() const {
  if (!hasSharedU())
    return -1;
  return indexSharedW() + shared_w.getDim();
}
