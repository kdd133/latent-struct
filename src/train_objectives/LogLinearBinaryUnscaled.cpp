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
#include "FeatureVector.h"
#include "Label.h"
#include "LogLinearBinaryUnscaled.h"
#include "LogWeight.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Utility.h"
#include "WeightVector.h"

void LogLinearBinaryUnscaled::valueAndGradientPart(const WeightVector& w,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, FeatureVector<RealWeight>& gradFv) {
  assert(gradFv.isDense() && !gradFv.isBinary());
  
  const WeightVector W0; // zero weight vector, for computing |Z(x)|
  
  const int d = w.getDim();
  const int ypos = TrainingObjective::kPositive;
  
  FeatureVector<LogWeight> feats(d, true);
  funcVal = 0;
  gradFv.zero();
  
  shared_array<RealWeight> tempVals(new RealWeight[d]); // passed to convert()
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == ypos) ? 1 : -1;
    
    // Compute the expected feature vector (normalized).
    //feats.zero(); // not needed because expectedFeatures performs reinit
    const LogWeight logMass = model.expectedFeatures(w, feats, xi, ypos, true);

    const LogWeight fW = (yi == 1) ? -logMass : logMass;
    funcVal += Utility::log1Plus(fW.convert()); // i.e., exp(fW)
    
    FeatureVector<RealWeight> realFeats = fvConvert(feats, tempVals, d);
    realFeats.timesEquals(-yi * (1 - Utility::sigmoid(-fW)));
    realFeats.addTo(gradFv);
  }
}

void LogLinearBinaryUnscaled::predictPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const WeightVector W0; // zero weight vector, for computing |Z(x)|
  const Label ypos = TrainingObjective::kPositive;  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    const LogWeight logMass = model.totalMass(w, x, ypos);
    const LogWeight logSizeZx = model.totalMass(W0, x, ypos);
    const LogWeight z = logMass * (-logSizeZx);
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, (-z));
  }
}
