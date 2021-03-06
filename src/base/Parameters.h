/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _PARAMETERS_H
#define _PARAMETERS_H

#include "Ublas.h"
#include "WeightVector.h"
#include <string>

// Parameters is a data structure that stores WeightVector objects w and u,
// which are understood to be concatenated as [w u]. Note that the u portion of
// the vector may be empty, as some training objectives use only w.

class Parameters {

public:
  WeightVector w;
  WeightVector u;
  WeightVector shared_w;
  WeightVector shared_u;
  
  Parameters() { init(); }
  Parameters(int dw) : w(dw) { init(); }
  Parameters(int dw, int du) : w(dw), u(du) { init(); }
  
  Parameters(const Parameters& other); // copy constructor
  Parameters& operator=(const Parameters& rhs); // assignment operator
  
  void add(const int index, const double v);
  
  std::size_t getDimW() const;
  std::size_t getDimU() const;
  
  // Return the number of w parameters plus the number of u parameters.
  std::size_t getDimWU() const;
  
  // The dimensionality of the gradient vector includes the shared parameters,
  // which are only the concern of the regularizer and are ignored by the
  // function that computes the objective value and gradient.
  std::size_t getDimTotal() const;
  
  bool hasU() const;
  
  bool hasSharedW() const;
  
  bool hasSharedU() const;
  
  double innerProd(const RealVec& fv) const;
  
  void setParams(const Parameters& other);
  
  void setWeights(const double* values, int len);
  
  double squaredL2Norm() const;
  
  void zero();
  
  double operator[](int index) const;
  
  int indexW() const;
  
  int indexU() const;
  
  int indexSharedW() const;
  
  int indexSharedU() const;
  
  void scale(double s);

private:
  void init();
  
  const static size_t _numWV = 4;
  
  WeightVector* _wv[_numWV];
};

#endif
