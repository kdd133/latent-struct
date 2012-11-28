/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "FeatureVector.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/lexical_cast.hpp>
#include <boost/shared_array.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
using namespace boost;
using namespace std;

WeightVector::WeightVector(int dim) : _weights(0) {
  reAlloc(dim);
}

WeightVector::WeightVector(shared_array<double> weights, int dim) :
    _weights(weights), _dim(dim) {
  assert(weights != 0);
  assert(_dim > 0);
  _l2 = 0;
  for (int i = 0; i < _dim; i++) {
    const double weight = _weights[i];
    _l2 += weight*weight;
  }
}

void WeightVector::reAlloc(int dim) {
  assert(dim > 0);
  _dim = dim;
  _weights.reset(new double[_dim]);
  zero();
}

double WeightVector::innerProd(const FeatureVector<RealWeight>& fv) const {
  if (_dim == 0)
    return 0.0;
  double prod = 0;
  for (int i = 0; i < fv.getNumEntries(); i++) {
    const int index = fv.getIndexAtLocation(i);
    assert(index >= 0 && index < _dim);
    prod += (fv.getValueAtLocation(i).toDouble() * _weights[index]);
  }
  return prod;
}

double WeightVector::innerProd(const FeatureVector<RealWeight>* fv) const {
  if (fv == 0)
    return 0.0;
  return innerProd(*fv);
}

// adapted from Parameters.h in egstra
void WeightVector::add(const FeatureVector<RealWeight>& fv, const double scale) {
  double l2Adjust = 0;
  if (fv.isBinary()) {
    for (int i = 0; i < fv.getNumEntries(); i++) {
      const int index = fv.getIndexAtLocation(i);
      assert(index >= 0 && index < _dim);
      const double currentVal = _weights[index];
      _weights[index] = currentVal + scale;
      /* use the formula: (w + x)^2 = w^2 + x^2 + 2*x*w */
      l2Adjust += scale * scale + 2 * scale * currentVal;
    }
  }
  else {
    for (int i = 0; i < fv.getNumEntries(); i++) {
      const int index = fv.getIndexAtLocation(i);
      assert(index >= 0 && index < _dim);
      const double update = scale * fv.getValueAtLocation(i).toDouble();
      const double currentVal = _weights[index];
      _weights[index] = currentVal + update;
      /* use the formula: (w + x)^2 = w^2 + x^2 + 2*x*w */
      l2Adjust += update * update + 2 * update * currentVal;
    }
  }
  _l2 += l2Adjust;
}

void WeightVector::add(const int index, const double update) {
  assert(!(index < 0 || index >= _dim));
  const double currentVal = _weights[index];
  _weights[index] += update;
  _l2 += update * update + 2 * update * currentVal;
}

void WeightVector::zero() {
  for (int i = 0; i < _dim; i++)
    _weights[i] = 0;
  _l2 = 0;
}

void WeightVector::setWeights(const double* newWeights, int len) {
  assert(len == _dim);
  assert(newWeights != _weights.get());
  _l2 = 0;
  for (int index = 0; index < _dim; index++) {
    const double update = newWeights[index];
    _weights[index] = update;
    _l2 += update * update;
  }
}

bool WeightVector::read(const string& fname, int dim) {
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

bool WeightVector::write(const string& fname) const {
  ofstream fout(fname.c_str());
  if (!fout.good())
    return false;
  for (int i = 0; i < _dim; i++)
    fout << i << " " << _weights[i] << endl;
  fout.close();
  return true;
}
