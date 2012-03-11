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
using namespace std;
#include "Dataset.h"
#include "Example.h"
#include "FeatureVector.h"
#include "Label.h"
#include "LogLinearBinaryObs.h"
#include "ObservedFeatureGen.h"
#include "Model.h"
#include "RealWeight.h"
#include "Utility.h"
#include "WeightVector.h"

void LogLinearBinaryObs::valueAndGradientPart(const WeightVector& w,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, FeatureVector<RealWeight>& gradFv) {
  assert(gradFv.isDense() && !gradFv.isBinary());
  
  funcVal = 0;
  gradFv.zero();
  
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == ypos) ? 1 : -1;
    
    bool own = false;
    FeatureVector<RealWeight>* phi = model.observedFeatures(xi, ypos, own);
    assert(phi);
    const double mass = -yi * w.innerProd(phi);
    funcVal += Utility::log1Plus(exp(mass));    
    phi->addTo(gradFv, -yi * (1 - Utility::sigmoid(-mass)));
    
    if (own) delete phi;
  }
}

void LogLinearBinaryObs::predictPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const Label ypos = TrainingObjective::kPositive;  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    FeatureVector<RealWeight>* phi = model.getFgenObserved()->getFeatures(x,
        ypos);
    assert(phi);
    const double z = w.innerProd(phi);
    delete phi;
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, -z);
  }
}
