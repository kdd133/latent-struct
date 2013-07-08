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
#include "LogLinearBinaryUnscaled.h"
#include "LogWeight.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Parameters.h"
#include "Ublas.h"
#include "Utility.h"
#include "WeightVector.h"

void LogLinearBinaryUnscaled::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  
  const int d = theta.w.getDim();
  const int ypos = TrainingObjective::kPositive;
  
  SparseLogVec feats(d);
  funcVal = 0;
  gradFv.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == ypos) ? 1 : -1;
    
    // Compute the expected feature vector (normalized).
    //feats.zero(); // not needed because expectedFeatures performs reinit
    const LogWeight logMass = model.expectedFeatures(theta.w, &feats, xi, ypos,
        true);

    const double fW = (yi == 1) ? -(double)logMass : (double)logMass;
    funcVal += Utility::log1Plus(exp(fW)); // i.e., exp(fW)
    
    ublas_util::addExponentiated(feats, gradFv, -yi*(1-Utility::sigmoid(-fW)));
  }
}

void LogLinearBinaryUnscaled::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const Label ypos = TrainingObjective::kPositive;  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    const LogWeight z = model.totalMass(theta.w, x, ypos);
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, (-z));
  }
}
