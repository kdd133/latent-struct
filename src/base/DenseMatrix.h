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

#include "SparseMatrix.h"
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
using boost::numeric::ublas::mapped_matrix;
using boost::numeric::ublas::matrix;

template <typename T>
class DenseMatrix {

public:
  DenseMatrix(int m, int n);

  void set(int row, int col, const T& v);
  
  T get(int row, int col) const;
  
  size_t numRows() const { return _M.size1(); }
  
  size_t numCols() const { return _M.size2(); }

  void plusEquals(const DenseMatrix<T>& M);
  
  void plusEquals(const SparseMatrix<T>& M);
  
  void timesEquals(const T& v);
  
  const matrix<T>& getMatrix() const { return _M; }
  
private:
  matrix<T> _M;
};

template <typename T>
DenseMatrix<T>::DenseMatrix(int m, int n) {
  assert(m > 0);
  assert(n > 0);
  _M = matrix<T>(m, n); // entries are initialized to 0
}

template <typename T>
void DenseMatrix<T>::set(int row, int col, const T& value) {
  assert(row < numRows());
  assert(col < numCols());
  _M(row, col) = value;
}

template <typename T>
T DenseMatrix<T>::get(int row, int col) const {
  assert(row < numRows());
  assert(col < numCols());
  return _M(row, col);
}

template <typename T>
void DenseMatrix<T>::plusEquals(const DenseMatrix<T>& M) {
  assert(M.numRows() == numRows());
  assert(M.numCols() == numCols());
  _M += M._M;
}

template <typename T>
void DenseMatrix<T>::plusEquals(const SparseMatrix<T>& sparse) {
  assert(numRows() == sparse.numRows());
  assert(numCols() == sparse.numCols());
  const mapped_matrix<T>& M = sparse.getMatrix();
  typename mapped_matrix<T>::const_iterator1 itRow;
  typename mapped_matrix<T>::const_iterator2 itCol;
  for (itRow = M.begin1(); itRow != M.end1(); ++itRow) {
    const int row = itRow.index1(); 
    for (itCol = itRow.begin(); itCol != itRow.end(); ++itCol) {
      const int col = itCol.index2();
      const T& toAdd = *itCol;
      _M(row, col) += toAdd;
    }
  }
}

template <typename T>
void DenseMatrix<T>::timesEquals(const T& scale) {
  _M *= scale;
}

#endif
