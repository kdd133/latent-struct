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
#include "FeatureVector.h"
#include "Label.h"
#include "MaxMarginMulti.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "RealWeight.h"
#include "Utility.h"
#include "WeightVector.h"
#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>
#include <vector>

void MaxMarginMulti::valueAndGradientPart(const WeightVector& w, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, FeatureVector<RealWeight>& gradFv) {
  assert(gradFv.isDense() && !gradFv.isBinary());
  
  const int d = w.getDim();
  
  std::vector<RealWeight> score(k, RealWeight());
  std::vector<FeatureVector<RealWeight> > feats(k, FeatureVector<RealWeight>(
      d, true));  
  
  funcVal = 0;
  gradFv.zero();
  
  assert(_imputedFv);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    RealWeight scoreMax(-numeric_limits<double>::infinity());
    Label yMax = 0;
    for (Label y = 0; y < k; y++) {
      score[y] = Utility::delta(yi,y) + model.maxFeatures(w, feats[y], xi, y);
      if (score[y] > scoreMax) {
        scoreMax = score[y];
        yMax = y;
      }
    }
    
    // Update the gradient and function value.
    feats[yMax].addTo(gradFv);    
    funcVal += score[yMax];
    
    // Subtract the observed features and score for the correct label yi.
    bool own = false;
    FeatureVector<RealWeight>* phi_yi = model.observedFeatures(xi, yi, own);
    assert(phi_yi);
    phi_yi->addTo(gradFv, -1);
    funcVal -= w.innerProd(phi_yi);
    if (own) delete phi_yi;
  }
}

void MaxMarginMulti::valueAndGradientFinalize(const WeightVector& w,
    double& funcVal, FeatureVector<RealWeight>& gradFv) {    
  // Subtract the sum of the imputed vectors from the gradient.
  _imputedFv->addTo(gradFv, -1.0);
  // Subtract the scores of the imputed vectors from the function value.
  funcVal = Utility::hinge(funcVal - w.innerProd(_imputedFv.get())); 
}

void MaxMarginMulti::setLatentFeatureVectorsPart(const WeightVector& w,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  const int d = w.getDim();
  FeatureVector<RealWeight> fv(d, true);  
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    // Note: The last argument is set to false because we wish to exclude the
    // observed features (if any) being included in the max feature vector.
    // Those do not need to be fixed in order to exploit the semi-convexity
    // property.
    model.maxFeatures(w, fv, xi, yi, false);
    boost::mutex::scoped_lock lock(_flag); // place a lock on _imputedFv
    fv.addTo(*_imputedFv.get());
  }
}

void MaxMarginMulti::initLatentFeatureVectors(const WeightVector& w) {
  assert(_dataset.numExamples() > 0);
  _imputedFv.reset(new FeatureVector<RealWeight>(w.getDim()));
}

void MaxMarginMulti::clearLatentFeatureVectors() {
  assert(_imputedFv);
  _imputedFv->zero();
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
