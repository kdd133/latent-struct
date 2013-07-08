/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include <boost/foreach.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "Dataset.h"
#include "Example.h"
#include "Label.h"
#include "LogLinearBinaryObs.h"
#include "ObservedFeatureGen.h"
#include "Model.h"
#include "Ublas.h"
#include "Utility.h"
#include "Parameters.h"

void LogLinearBinaryObs::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  
  funcVal = 0;
  gradFv.clear();
  
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == ypos) ? 1 : -1;
    
    bool own = false;
    SparseRealVec* phi = model.observedFeatures(xi, ypos, own);
    assert(phi);
    const double mass = -yi * theta.w.innerProd(*phi);
    funcVal += Utility::log1Plus(exp(mass));
    gradFv += (*phi) * (-yi * (1 - Utility::sigmoid(-mass)));
    
    if (own) delete phi;
  }
}

void LogLinearBinaryObs::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const Label ypos = TrainingObjective::kPositive;  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    bool own = false;
    SparseRealVec* phi = model.observedFeatures(x, ypos, own);
    assert(phi);
    const double z = theta.w.innerProd(*phi);
    if (own) delete phi;
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, -z);
  }
}
