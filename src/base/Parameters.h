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
  
  Parameters() { }
  Parameters(int dw) : w(dw) { }
  Parameters(int dw, int du) : w(dw), u(du) { }
  
  void add(const int index, const double v);
  
  // Return the "total dimensionality" of the parameters; i.e., the sum of the
  // dimensionalities of the component vectors.
  std::size_t getTotalDim() const;
  
  bool hasU() const;
  
  double innerProd(const RealVec& fv) const;
  
  void setParams(const Parameters& other);
  
  void setWeights(const double* values, int len);
  
  double squaredL2Norm() const;
  
  void zero();
  
  const double& operator[](int index) const;
};

#endif
