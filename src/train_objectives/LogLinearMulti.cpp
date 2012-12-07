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
#include "Ublas.h"
#include "WeightVector.h"
#include <boost/shared_array.hpp>
#include <vector>

void LogLinearMulti::valueAndGradientPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, RealVec& gradFv) {
  
  const int d = w.getDim();
  
  std::vector<LogWeight> mass(k, LogWeight());
  std::vector<LogVec> feats(k, LogVec(d));  
  LogVec featsTotal(d);
  RealVec temp(d);
  
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
      mass[y] = model.expectedFeatures(w, feats[y], xi, y, false);
      featsTotal += feats[y];
      massTotal += mass[y];
    }
    
    // Normalize
    featsTotal *= -massTotal;
    feats[yi] *= -mass[yi];

    // Convert features from log- to real-space, then update gradient
    gradFv += ublas_util::convertVec(featsTotal, temp);
    gradFv -= ublas_util::convertVec(feats[yi], temp);

    // Update function value
    // Note: We want the log, which is why we don't convert to RealWeight.
    funcVal += massTotal;
    funcVal -= mass[yi];
  }
}

void LogLinearMulti::predictPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      const double yScore = model.totalMass(w, x, y);
      scores.setScore(id, y, yScore);
    }
  }
}
