/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _FEATUREMATRIX_H
#define _FEATUREMATRIX_H

#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
using boost::numeric::ublas::matrix;

class Alphabet;
class LogWeight;

class FeatureMatrix {

public:
  FeatureMatrix(int m);

  void set(int row, int col, const LogWeight v);
  
  LogWeight get(int row, int col) const;

  void append(const FeatureMatrix& toAppend);
  
  void timesEquals(const LogWeight& v);
  
  void print(std::ostream& out, const Alphabet& alphabet);
  
  void print(std::ostream& out);
  
private:
  matrix<LogWeight> _A;
};

#endif
