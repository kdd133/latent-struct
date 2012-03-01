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
#include "MaxMarginBinaryObs.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "RealWeight.h"
#include "Utility.h"
#include "WeightVector.h"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

void MaxMarginBinaryObs::valueAndGradientPart(const WeightVector& w,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, FeatureVector<RealWeight>& gradFv) {
  assert(gradFv.isDense() && !gradFv.isBinary());
  
  funcVal = 0;
  gradFv.zero();
  
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == TrainingObjective::kPositive) ? 1 : -1;
    
    bool own = false;
    FeatureVector<RealWeight>* phi = model.observedFeatures(xi, ypos, own);
    const double z = yi * w.innerProd(phi);
    funcVal += Utility::hinge(1-z);
    
    // Gradient contribution is 0 if z=y*w'*phi >= 1, and -y*phi otherwise. 
    if (z < 1)
      phi->addTo(gradFv, -yi);

    if (own && !phi->release()) delete phi;
  }
}

void MaxMarginBinaryObs::predictPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const Label ypos = TrainingObjective::kPositive;
  boost::shared_ptr<ObservedFeatureGen> fgen = model.getFgenObserved();
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    FeatureVector<RealWeight>* phi = fgen->getFeatures(x, ypos);
    const double z = w.innerProd(phi);
    if (!phi->release()) delete phi;
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, -z);
  }
}
