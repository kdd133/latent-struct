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
#include <boost/lexical_cast.hpp>
#include <boost/shared_array.hpp>
#include <boost/tokenizer.hpp>
#include <cmath>
#include <fstream>

using namespace boost;

WeightVector::WeightVector(int dim) : _scale(1) {
  reAlloc(dim);
}

WeightVector::WeightVector(shared_array<double> weights, int dim) :
    _weights(weights), _dim(dim), _scale(1) {
  assert(weights != 0);
  assert(_dim > 0);
}

void WeightVector::reAlloc(int dim) {
  _dim = dim;
  if (dim > 0)
    _weights.reset(new double[_dim]);
  else
    _weights.reset();
  zero();
}

double WeightVector::innerProd(const SparseLogVec& fv) const {
  if (_dim == 0 || fv.size() == 0)
    return 0.0;
  double prod = 0;
  SparseLogVec::const_iterator it;
  for (it = fv.begin(); it != fv.end(); ++it) {
    assert(it.index() < _dim);
    prod += exp(*it) * _weights[it.index()];
  }
  return _scale * prod;
}

double WeightVector::innerProd(const SparseRealVec& fv) const {
  if (_dim == 0 || fv.size() == 0)
    return 0.0;
  double prod = 0;
  SparseRealVec::const_iterator it;
  for (it = fv.begin(); it != fv.end(); ++it) {
    assert(it.index() < _dim);
    prod += (*it) * _weights[it.index()];
  }
  return _scale * prod;
}

double WeightVector::innerProd(const LogVec& fv) const {
  assert(fv.size() == _dim);
  if (_dim == 0)
    return 0.0;
  double prod = 0;
  for (size_t i = 0; i < fv.size(); ++i)
    prod += exp(fv(i)) * _weights[i];
  return _scale * prod;
}

double WeightVector::innerProd(const RealVec& fv) const {
  assert(fv.size() == _dim);
  if (_dim == 0)
    return 0.0;
  double prod = 0;
  for (size_t i = 0; i < fv.size(); ++i)
    prod += fv(i) * _weights[i];
  return _scale * prod;
}

void WeightVector::add(const int index, const double update) {
  assert(!(index < 0 || index >= _dim));
  _weights[index] += update / _scale;
}

void WeightVector::zero() {
  for (int i = 0; i < _dim; i++)
    _weights[i] = 0;
  _scale = 1;
}

void WeightVector::setWeights(const double* rawWeights, int len) {
  assert(rawWeights != _weights.get());
  if (len !=  _dim)
    reAlloc(len);
  for (int index = 0; index < _dim; index++)
    _weights[index] = rawWeights[index];
  _scale = 1;
}

void WeightVector::setWeights(const WeightVector& w) {
  if (this != &w) {
    if (w._dim !=  _dim)
      reAlloc(w._dim);
    for (int index = 0; index < _dim; index++)
      _weights[index] = w._weights[index];
    _scale = w._scale;
  }
}

bool WeightVector::read(const std::string& fname, int dim) {
  using namespace std;
  reAlloc(dim);
  ifstream fin(fname.c_str(), ifstream::in);
  if (!fin.good())
    return false;
  char_separator<char> spaceSep(" ");
  string line;
  while (getline(fin, line)) {
    tokenizer<char_separator<char> > fields(line, spaceSep);
    tokenizer<char_separator<char> >::const_iterator it = fields.begin();
    int index = lexical_cast<int>(*it++);
    double value = lexical_cast<double>(*it++);
    assert(it == fields.end());
    add(index, value);
  }
  fin.close();
  return true;
}

bool WeightVector::write(const std::string& fname) const {
  using namespace std;
  ofstream fout(fname.c_str());
  if (!fout.good())
    return false;
  for (int i = 0; i < _dim; i++)
    fout << i << " " << _scale * _weights[i] << endl;
  fout.close();
  return true;
}

std::ostream& operator<<(std::ostream& out, const WeightVector& w) {
  out << "[" << w._dim << "](";
  if (w._dim == 0) {
    out << ")";
    return out;
  }
  for (int index = 0; index < w._dim - 1; index++)
    out << w._scale * w._weights[index] << ",";
  out << w._scale * w._weights[w._dim - 1] << ")";
  return out;
}

double WeightVector::operator[](int index) const {
  assert(index < _dim);
  return _weights[index] * _scale;
}

void WeightVector::scale(const double s) {
  _scale *= s;
  if (_scale < 1e-5)
    rescale();
}

void WeightVector::rescale() {
  if (_scale != 1.0) {
    for (int i = 0; i < _dim; i++) {
      const double v = _weights[i] * _scale;
      _weights[i] = v;
    }
    _scale = 1;
  }
}

double WeightVector::squaredL2Norm() const {
  double prod = 0;
  for (int i = 0; i < _dim; i++)
    prod += _weights[i] * _weights[i];
  return _scale * _scale * prod;
}

double WeightVector::getScale() const {
  return _scale;
}
