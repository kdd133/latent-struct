/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Parameters.h"
#include "RegularizerL2.h"
#include "Ublas.h"
#include <assert.h>

void RegularizerL2::addRegularization(const Parameters& theta, double& fval,
    RealVec& grad) const {
  assert(_beta > 0.0);
  const int d = theta.getTotalDim();
  assert(d == grad.size());
  fval += _beta/2 * theta.squaredL2Norm();
  for (size_t i = 0; i < d; ++i)
    grad(i) += theta.getWeight(i) * _beta; // add beta*theta to gradient
}
