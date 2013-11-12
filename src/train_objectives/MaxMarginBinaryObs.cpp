/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Dataset.h"
#include "Example.h"
#include "Label.h"
#include "MaxMarginBinaryObs.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Parameters.h"
#include "Ublas.h"
#include "Utility.h"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace boost;

void MaxMarginBinaryObs::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  
  funcVal = 0;
  assert(!theta.hasU()); // There should be no latent variables.
  
  // It is faster to accumulate using a dense vector.
  RealVec gradDense(theta.w.getDim());
  gradDense.clear();
  
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == TrainingObjective::kPositive) ? 1 : -1;
    
    shared_ptr<const SparseRealVec> phi = model.observedFeatures(xi, ypos);
    assert(phi);
    const double z = yi * theta.w.innerProd(*phi);
    funcVal += Utility::hinge(1-z);
    
    // Gradient contribution is 0 if z=y*w'*phi >= 1, and -y*phi otherwise. 
    if (z < 1)
      noalias(gradDense) += -yi * (*phi);
  }
  noalias(gradFv) = gradDense;
}

void MaxMarginBinaryObs::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    shared_ptr<const SparseRealVec> phi = model.observedFeatures(x, ypos);
    assert(phi);
    const double z = theta.w.innerProd(*phi);
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, -z);
  }
}
