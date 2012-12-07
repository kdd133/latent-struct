/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentFeatureGen.h"
#include "Dataset.h"
#include "Example.h"
#include "Label.h"
#include "MaxMarginMulti.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Ublas.h"
#include "Utility.h"
#include "WeightVector.h"
#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>
#include <vector>

using namespace std;

void MaxMarginMulti::valueAndGradientPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, RealVec& gradFv) {
  
  const int d = w.getDim();
  
  vector<double> score(k);
  vector<SparseRealVec> feats(k, SparseRealVec(d));  
  
  funcVal = 0;
  gradFv.clear();
  
  // This if statement should never be executed while the objective is being
  // optimized, since the (EM) optimizer will perform the initialization.
  // However, if the gradient is requested outside of the optimization routine,
  // we need to initialize the latent FVs using the given weight vector.
  if (!_imputedFv)
    initLatentFeatureVectors(w);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    double scoreMax(-numeric_limits<double>::infinity());
    Label yMax = 0;
    for (Label y = 0; y < k; y++) {
      score[y] = Utility::delta(yi,y) + model.maxFeatures(w, feats[y], xi, y);
      if (score[y] > scoreMax) {
        scoreMax = score[y];
        yMax = y;
      }
    }
    
    // Update the gradient and function value.
    gradFv += feats[yMax];
    funcVal += score[yMax];
    
    // Subtract the observed features and score for the correct label yi.
    bool own = false;
    SparseRealVec* phi_yi = model.observedFeatures(xi, yi, own);
    assert(phi_yi);
    gradFv -= (*phi_yi);
    funcVal -= w.innerProd(*phi_yi);
    if (own) delete phi_yi;
  }
}

void MaxMarginMulti::valueAndGradientFinalize(const WeightVector& w,
    double& funcVal, RealVec& gradFv) {    
  // Subtract the sum of the imputed vectors from the gradient.
  gradFv -= (*_imputedFv);
  // Subtract the scores of the imputed vectors from the function value.
  funcVal = Utility::hinge(funcVal - w.innerProd(*_imputedFv)); 
}

void MaxMarginMulti::setLatentFeatureVectorsPart(const WeightVector& w,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  const int d = w.getDim();
  SparseRealVec fv(d);  
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    // Note: The last argument is set to false because we wish to exclude the
    // observed features (if any) being included in the max feature vector.
    // Those do not need to be fixed in order to exploit the semi-convexity
    // property.
    model.maxFeatures(w, fv, xi, yi, false);
    boost::mutex::scoped_lock lock(_flag); // place a lock on _imputedFv
    (*_imputedFv) += fv;
  }
}

void MaxMarginMulti::initLatentFeatureVectors(const WeightVector& w) {
  assert(_dataset.numExamples() > 0);
  _imputedFv.reset(new SparseRealVec(w.getDim()));
}

void MaxMarginMulti::clearLatentFeatureVectors() {
  assert(_imputedFv);
  _imputedFv->clear();
}

void MaxMarginMulti::predictPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      const double yScore = model.viterbiScore(w, x, y);
      scores.setScore(id, y, yScore);
    }
  }
}
