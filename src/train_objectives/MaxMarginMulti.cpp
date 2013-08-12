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
#include "Parameters.h"
#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>
#include <vector>

using namespace std;

void MaxMarginMulti::valueAndGradientPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  
  const int d = theta.w.getDim();
  
  vector<double> score(k);
  vector<SparseRealVec> feats(k, SparseRealVec(d));  
  
  funcVal = 0;
  
  // It is faster to accumulate using a dense vector.
  RealVec gradDense(d);
  gradDense.clear();
  
  // This if statement should never be executed while the objective is being
  // optimized, since the (EM) optimizer will perform the initialization.
  // However, if the gradient is requested outside of the optimization routine,
  // we need to initialize the latent FVs using the given weight vector.
  if (!_imputedFv)
    initLatentFeatureVectors(theta);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    double scoreMax(-numeric_limits<double>::infinity());
    Label yMax = 0;
    for (Label y = 0; y < k; y++) {
      score[y] = Utility::delta(yi,y) + model.maxFeatures(theta.w, &feats[y],
          xi, y);
      if (score[y] > scoreMax) {
        scoreMax = score[y];
        yMax = y;
      }
    }
    
    // Update the gradient and function value.
    noalias(gradDense) += feats[yMax];
    funcVal += score[yMax];
    
    // Subtract the observed features and score for the correct label yi.
    bool own = false;
    SparseRealVec* phi_yi = model.observedFeatures(xi, yi, own);
    assert(phi_yi);
    noalias(gradDense) -= (*phi_yi);
    funcVal -= theta.w.innerProd(*phi_yi);
    if (own) delete phi_yi;
  }
  noalias(gradFv) = gradDense;
}

void MaxMarginMulti::valueAndGradientFinalize(const Parameters& theta,
    double& funcVal, SparseRealVec& gradFv) {    
  // Subtract the sum of the imputed vectors from the gradient.
  noalias(gradFv) -= (*_imputedFv);
  // Subtract the scores of the imputed vectors from the function value.
  funcVal = Utility::hinge(funcVal - theta.w.innerProd(*_imputedFv)); 
}

void MaxMarginMulti::setLatentFeatureVectorsPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  const int d = theta.w.getDim();
  SparseRealVec fv(d);  
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    // Note: The last argument is set to false because we wish to exclude the
    // observed features (if any) being included in the max feature vector.
    // Those do not need to be fixed in order to exploit the semi-convexity
    // property.
    model.maxFeatures(theta.w, &fv, xi, yi, false);
    boost::mutex::scoped_lock lock(_flag); // place a lock on _imputedFv
    (*_imputedFv) += fv;
  }
}

void MaxMarginMulti::initLatentFeatureVectors(const Parameters& theta) {
  assert(_dataset.numExamples() > 0);
  _imputedFv.reset(new SparseRealVec(theta.w.getDim()));
}

void MaxMarginMulti::clearLatentFeatureVectors() {
  assert(_imputedFv);
  _imputedFv->clear();
}

void MaxMarginMulti::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      const double yScore = model.viterbiScore(theta.w, x, y);
      scores.setScore(id, y, yScore);
    }
  }
}
