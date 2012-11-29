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
  _A = matrix<LogWeight>(m, m); // entries are initialized to 0
}

void FeatureMatrix::set(int row, int col, LogWeight value) {
  assert(row < _A.size1());
  assert(col < _A.size2());
  _A(row, col) = value;
}

LogWeight FeatureMatrix::get(int row, int col) const {
  assert(row < _A.size1());
  assert(col < _A.size2());
  return _A(row, col);
}

void FeatureMatrix::append(const FeatureMatrix& fm) {
  assert(fm._A.size1() == _A.size1());
  assert(fm._A.size2() == _A.size2());
  _A += fm._A;
}

void FeatureMatrix::timesEquals(const LogWeight& value) {
  _A *= value;
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
