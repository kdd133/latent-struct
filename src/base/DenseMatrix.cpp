/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "DenseMatrix.h"
#include "LogWeight.h"
#include <boost/numeric/ublas/matrix.hpp>
using boost::numeric::ublas::matrix;

DenseMatrix::DenseMatrix(int m) {
  assert(m > 0);
  _A = matrix<LogWeight>(m, m); // entries are initialized to 0
}

void DenseMatrix::set(int row, int col, const LogWeight& value) {
  assert(row < numRows());
  assert(col < numCols());
  _A(row, col) = value;
}

LogWeight DenseMatrix::get(int row, int col) const {
  assert(row < numRows());
  assert(col < numCols());
  return _A(row, col);
}

void DenseMatrix::plusEquals(const DenseMatrix& fm) {
  assert(fm.numRows() == numRows());
  assert(fm.numCols() == numCols());
  _A += fm._A;
}

void DenseMatrix::timesEquals(const LogWeight& value) {
  _A *= value;
}
