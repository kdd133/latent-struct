/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _DENSEMATRIX_H
#define _DENSEMATRIX_H

#include <boost/numeric/ublas/matrix.hpp>
using boost::numeric::ublas::matrix;

template <typename T>
class DenseMatrix {

public:
  DenseMatrix(int m);

  void set(int row, int col, const T& v);
  
  T get(int row, int col) const;
  
  size_t numRows() const { return _A.size1(); }
  
  size_t numCols() const { return _A.size2(); }

  void plusEquals(const DenseMatrix& toAppend);
  
  void timesEquals(const T& v);
  
private:
  matrix<T> _A;
};

template <typename T>
DenseMatrix<T>::DenseMatrix(int m) {
  assert(m > 0);
  _A = matrix<T>(m, m); // entries are initialized to 0
}

template <typename T>
void DenseMatrix<T>::set(int row, int col, const T& value) {
  assert(row < numRows());
  assert(col < numCols());
  _A(row, col) = value;
}

template <typename T>
T DenseMatrix<T>::get(int row, int col) const {
  assert(row < numRows());
  assert(col < numCols());
  return _A(row, col);
}

template <typename T>
void DenseMatrix<T>::plusEquals(const DenseMatrix<T>& fm) {
  assert(fm.numRows() == numRows());
  assert(fm.numCols() == numCols());
  _A += fm._A;
}

template <typename T>
void DenseMatrix<T>::timesEquals(const T& value) {
  _A *= value;
}

#endif
