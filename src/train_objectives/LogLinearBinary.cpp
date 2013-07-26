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
#include "LogLinearBinary.h"
#include "LogWeight.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Parameters.h"
#include "Ublas.h"
#include "Utility.h"
#include "WeightVector.h"
#include <boost/shared_array.hpp>
#include <cmath>

using boost::shared_array;

LogLinearBinary::LogLinearBinary(const Dataset& dataset,
    const std::vector<Model*>& models) : TrainingObjective(dataset, models) {
  Parameters theta; // dummy: not used by setLatentFeatureVectors here
  setLatentFeatureVectors(theta);
}

void LogLinearBinary::valueAndGradientPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  
  const int d = theta.w.getDim();
  SparseLogVec feats(d);
  funcVal = 0;
  gradFv.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == TrainingObjective::kPositive) ? 1 : -1;
    
    // Compute the expected feature vector (normalized).
    //feats.zero(); // not needed because expectedFeatures performs reinit
    const LogWeight logMass = model.expectedFeatures(theta.w, &feats, xi,
        TrainingObjective::kPositive, true);

    // Lookup the pre-computed value of logSizeZx for this xi.
    // (it was computed in setLatentFeatureVectorsPart below)
    LogWeight logSizeZx;
    const DictType::const_iterator item = _logSizeZxMap.find(xi.getId());
    assert(item != _logSizeZxMap.end());
    logSizeZx = item->second;

    const LogWeight z = logMass / logSizeZx;
    const double fW = (yi == 1) ? -(double)z : (double)z;
    funcVal += Utility::log1Plus(exp(fW));
    
    ublas_util::addExponentiated(feats, gradFv, -yi*(1-Utility::sigmoid(-fW)));
  }
}

void LogLinearBinary::setLatentFeatureVectorsPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  const WeightVector W0; // zero weight vector
  for (Dataset::iterator it = begin; it != end; ++it) {
    // Compute the number of paths through the fst using the zero weight vector.
    // We cache this value in a lookup table (implemented as a map). This is
    // possible because the value does not depend on the current weight vector.
    const Pattern& xi = *it->x();
    LogWeight logSizeZx = model.totalMass(W0, xi, TrainingObjective::kPositive);
    boost::mutex::scoped_lock lock(_logSizeZxMapLock);
    _logSizeZxMap.insert(PairType(xi.getId(), logSizeZx));
  }
}

void LogLinearBinary::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const WeightVector W0; // zero weight vector, for computing |Z(x)|
  const Label ypos = TrainingObjective::kPositive;  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    const LogWeight logMass = model.totalMass(theta.w, x, ypos);
    const LogWeight logSizeZx = model.totalMass(W0, x, ypos);
    const LogWeight z = logMass / logSizeZx;
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, (-z));
  }
}
