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

void LogLinearBinary::valueAndGradientPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, RealVec& gradFv) {
  
  const WeightVector W0; // zero weight vector, for computing |Z(x)|
  
  const int d = theta.w.getDim();
  const int ypos = TrainingObjective::kPositive;
  
  LogVec feats(d);
  RealVec temp(d);
  funcVal = 0;
  gradFv.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == ypos) ? 1 : -1;
    
    // Compute the expected feature vector (normalized).
    //feats.zero(); // not needed because expectedFeatures performs reinit
    const LogWeight logMass = model.expectedFeatures(theta.w, feats, xi, ypos,
        true);
    
    // Compute the number of paths through the fst using the zero weight vector.
    // We cache this value in a lookup table (implemented as a map). This is
    // possible because the value does not depend on the current weight vector.
    // Note that, since the map for each thread will only have entries for the
    // patter ids in its particular partition of the Dataset, no explicit
    // syncronization is needed.
    LogWeight logSizeZx;
    const DictType::const_iterator item = _logSizeZxMap.find(xi.getId());
    if (item == _logSizeZxMap.end()) {
      logSizeZx = model.totalMass(W0, xi, ypos);
      std::pair<DictType::iterator, bool> ret = _logSizeZxMap.insert(
          PairType(xi.getId(), logSizeZx));
      assert(ret.second); // will be false if entry already present in map
    }
    else
      logSizeZx = item->second;

    const LogWeight z = logMass / logSizeZx;
    const double fW = (yi == 1) ? -(double)z : (double)z;
    funcVal += Utility::log1Plus(exp(fW));
    
    ublas_util::convertVec(feats, temp);
    gradFv += (temp * -yi * (1 - Utility::sigmoid(-fW)));
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
