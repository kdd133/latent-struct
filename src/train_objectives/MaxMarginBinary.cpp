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
#include "MaxMarginBinary.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Ublas.h"
#include "Utility.h"
#include "Parameters.h"
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_ptr.hpp>

using namespace boost;

void MaxMarginBinary::valueAndGradientPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  
  const int d = theta.w.getDim();
  const int ypos = TrainingObjective::kPositive;
  
  SparseRealVec feats(d);
  funcVal = 0;
  
  // It is faster to accumulate using a dense vector.
  RealVec gradDense(d);
  gradDense.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == ypos) ? 1 : -1;
    
    // Note: Gradient contribution is 0 if z >= 1, and -y*feats otherwise.
    double z;
    if (yi == -1) {
      feats.clear();
      // gets latent+obs feats
      z = yi * model.maxFeatures(theta.w, &feats, xi, ypos);
      if (z < 1)
        noalias(gradDense) += -yi * feats;
    }
    else {
      const size_t i = xi.getId();
      assert(_imputedFvs.size() > i);
      // Since we did not fix the observed features in setLatentFeatureVectorsPart,
      // we need to factor them in here.
      shared_ptr<const SparseRealVec> phiObs = model.observedFeatures(xi, ypos);
      assert(phiObs);
      z = yi * (theta.w.innerProd(*phiObs) + theta.w.innerProd(_imputedFvs[i]));
      if (z < 1) {
        noalias(gradDense) += -yi * (_imputedFvs[i]);
        noalias(gradDense) += -yi * (*phiObs);
      }
    }
    funcVal += Utility::hinge(1 - z);
  }
  noalias(gradFv) = gradDense;
}

void MaxMarginBinary::setLatentFeatureVectorsPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = (it->y() == ypos) ? 1 : -1;    
    if (yi == 1) {
      const size_t i = xi.getId();
      assert(_imputedFvs.size() > i);
      // The last argument in the call to maxFeatures is false because we do
      // not want to fix the observed features when computing the objective.
      model.maxFeatures(theta.w, &_imputedFvs[i], xi, yi, false);
    }
  }
}

void MaxMarginBinary::initLatentFeatureVectors(const Parameters& theta) {
  assert(_dataset.numExamples() > 0);
  
  // If we sub-sampled the dataset, there is no longer a 1-to-1 correspondence
  // between ids and indices, so we need to know the maximum id that is present.
  // (Or, we could use a ptr_map, but that would hurt time efficiency.)
  const size_t maxId = _dataset.getMaxId();
  
  for (size_t i = 0; i <= maxId; i++)
    _imputedFvs.push_back(new SparseRealVec());
  
  const Label ypos = TrainingObjective::kPositive;
  const int d = theta.w.getDim();
  BOOST_FOREACH(const Example& ex, _dataset.getExamples()) {
    const size_t id = ex.x()->getId();
    if (ex.y() == ypos)
      _imputedFvs[id].resize(d);
  }
}

void MaxMarginBinary::clearLatentFeatureVectors() {
  // Do nothing. FeatureVectors will be overwritten by calls to maxFeatures()
  // in setLatentFeatureVectorsPart().
}

MaxMarginBinary::~MaxMarginBinary() {
  _imputedFvs.clear();
}

void MaxMarginBinary::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    const double z = model.viterbiScore(theta.w, x, ypos);
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, (-z));
  }
}
