/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include <boost/foreach.hpp>
#include <boost/shared_array.hpp>
using boost::shared_array;
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;
#include "Dataset.h"
#include "Example.h"
#include "FeatureVector.h"
#include "Label.h"
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "Model.h"
#include "RealWeight.h"
#include "WeightVector.h"

void LogLinearMulti::valueAndGradientPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, FeatureVector<RealWeight>& gradFv) {
  assert(gradFv.isDense() && !gradFv.isBinary());
  
  const int d = w.getDim();
  
  vector<LogWeight> mass(k, LogWeight());
  vector<FeatureVector<LogWeight> > feats(k, FeatureVector<LogWeight>(d, true));  
  FeatureVector<LogWeight> featsTotal(d);
  shared_array<RealWeight> tempVals(new RealWeight[d]); // passed to convert()
  
  funcVal = 0;
  gradFv.zero();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    LogWeight massTotal(LogWeight::kZero);

    // Note: feats[y] is initially zeroed out by expectedFeatures() 
    featsTotal.zero();
    
    for (Label y = 0; y < k; y++) {
      // Note: The last argument is false b/c we want unnormalized features.
      mass[y] = model.expectedFeatures(w, feats[y], xi, y, false);
      feats[y].addTo(featsTotal);
      massTotal.plusEquals(mass[y]);
    }
    
    // Normalize
    featsTotal.timesEquals(-massTotal);
    feats[yi].timesEquals(-mass[yi]);

    // Convert features from log- to real-space, then update gradient
    convert(featsTotal, tempVals, d).addTo(gradFv);
    convert(feats[yi], tempVals, d).addTo(gradFv, -1.0);

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
