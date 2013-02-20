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
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "Model.h"
#include "Parameters.h"
#include "Ublas.h"
#include <boost/shared_array.hpp>
#include <vector>

void LogLinearMulti::valueAndGradientPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, RealVec& gradFv) {
  
  const int d = theta.w.getDim();
  
  std::vector<LogWeight> mass(k, LogWeight());
  std::vector<SparseLogVec> feats(k, LogVec(d));  
  SparseLogVec featsTotal(d);
  
  funcVal = 0;
  gradFv.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    LogWeight massTotal(0);

    // Note: feats[y] is initially zeroed out by expectedFeatures() 
    featsTotal.clear();
    
    for (Label y = 0; y < k; y++) {
      // Note: The last argument is false b/c we want unnormalized features.
      mass[y] = model.expectedFeatures(theta.w, &feats[y], xi, y, false);
      featsTotal += feats[y];
      massTotal += mass[y];
    }
    
    // Normalize
    featsTotal /= massTotal;
    feats[yi] /= mass[yi];

    // Convert features from log- to real-space, then update gradient
    ublas_util::addExponentiated(featsTotal, gradFv, 1);
    ublas_util::addExponentiated(feats[yi], gradFv, -1);

    // Update function value
    // Note: We want the log, which is why we don't exponentiate.
    funcVal += (double)massTotal - (double)mass[yi];
  }
}

void LogLinearMulti::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      const double yScore = model.totalMass(theta.w, x, y);
      scores.setScore(id, y, yScore);
    }
  }
}
