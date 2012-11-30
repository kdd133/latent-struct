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

class LogWeight;

class DenseMatrix {

public:
  DenseMatrix(int m);

  void set(int row, int col, const LogWeight& v);
  
  LogWeight get(int row, int col) const;
  
  size_t numRows() const { return _A.size1(); }
  
  size_t numCols() const { return _A.size2(); }

  void plusEquals(const DenseMatrix& toAppend);
  
  void timesEquals(const LogWeight& v);
  
private:
  matrix<LogWeight> _A;
};

#endif
