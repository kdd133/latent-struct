/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Alphabet.h"
#include "FeatureMatrix.h"
#include "LogWeight.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
using boost::numeric::ublas::matrix;

FeatureMatrix::FeatureMatrix(int m) {
  assert(m > 0);
  _A = matrix<double>(m, m);
  
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < m; ++j)
      _A(i,j) = LogWeight::kZero;
}

void FeatureMatrix::assign(int row, int col, double value) {
  assert(row < _A.size1());
  assert(col < _A.size2());
  _A(row, col) = value;
}

double FeatureMatrix::get(int row, int col) const {
  assert(row < _A.size1());
  assert(col < _A.size2());
  return _A(row, col);
}

void FeatureMatrix::logAppend(const FeatureMatrix& fm) {
  assert(fm._A.size1() == _A.size1());
  assert(fm._A.size2() == _A.size2());
  for (size_t i = 0; i < _A.size1(); ++i) {
    for (size_t j = 0; j < _A.size2(); ++j) {
      LogWeight dest(get(i,j));
      LogWeight append(fm.get(i,j));
      dest += append;
      _A(i,j) = dest;
    }
  }
}

void FeatureMatrix::timesEquals(double value) {
  for (size_t i = 0; i < _A.size1(); ++i) {
    for (size_t j = 0; j < _A.size2(); ++j) {
      LogWeight current(_A(i,j));
      LogWeight scale(value);
      current *= scale;
      _A(i,j) = current;
    }
  }
}

void FeatureMatrix::print(std::ostream& out, const Alphabet& alphabet) {
  for (int i = 0; i < _A.size1(); ++i)
    for (int j = 0; j < _A.size2(); ++j)
      out << alphabet.reverseLookup(i) << "," << alphabet.reverseLookup(j)
          << "\t" << get(i,j) << endl;
}

void FeatureMatrix::print(std::ostream& out) {
  for (int i = 0; i < _A.size1(); ++i) {
    for (int j = 0; j < _A.size2(); ++j)
      out << "(" << i << "," << j << ")" << get(i,j) << "\t";
    out << endl;
  }
}