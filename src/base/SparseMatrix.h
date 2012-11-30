/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _SPARSEMATRIX_H
#define _SPARSEMATRIX_H

#include <boost/numeric/ublas/matrix_sparse.hpp>
using boost::numeric::ublas::mapped_matrix;

template <typename T>
class SparseMatrix {
  
public:
  SparseMatrix(int m, int n);

  void set(int row, int col, const T& v);
  
  T get(int row, int col) const;
  
  size_t numRows() const { return _M.size1(); }
  
  size_t numCols() const { return _M.size2(); }

  void plusEquals(const SparseMatrix<T>& M);
  
  void timesEquals(const T& v);
  
  const mapped_matrix<T>& getMatrix() const { return _M; }
  
private:
  mapped_matrix<T> _M;
};

template <typename T>
SparseMatrix<T>::SparseMatrix(int m, int n) {
  assert(m > 0);
  assert(n > 0);
  _M = mapped_matrix<T>(m, n); // entries are initialized to 0
}

template <typename T>
void SparseMatrix<T>::set(int row, int col, const T& value) {
  assert(row < numRows());
  assert(col < numCols());
  _M(row, col) = value;
}

template <typename T>
T SparseMatrix<T>::get(int row, int col) const {
  assert(row < numRows());
  assert(col < numCols());
  return _M(row, col);
}

template <typename T>
void SparseMatrix<T>::plusEquals(const SparseMatrix<T>& M) {
  assert(M.numRows() == numRows());
  assert(M.numCols() == numCols());
  _M += M._M;
}

template <typename T>
void SparseMatrix<T>::timesEquals(const T& scale) {
  _M *= scale;
}

#endif
