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
#include "LogLinearMultiELFV.h"
#include "LogWeight.h"
#include "Model.h"
#include "Parameters.h"
#include "Ublas.h"
#include <assert.h>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <cmath>
#include <vector>

void LogLinearMultiELFV::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, RealVec& gradFv) {

  const int n = theta.w.getDim();    // i.e., the number of features
  const int d = theta.getTotalDim(); // i.e., the length of the [w u] vector
  assert(theta.hasU());
  
  std::vector<RealVec> feats(k, LogVec(n));
  LogVec logFeats; // call to expectedFeatures will allocate/resize
  RealVec featsTotal(d);
  double mass_y;
  double massTotal;
  
  funcVal = 0;
  gradFv.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    massTotal = 0;
    featsTotal.clear();
    
    for (Label y = 0; y < k; y++) {      
      // Note: The last argument is true b/c we want normalized features.
      model.expectedFeatures(theta.u, logFeats, xi, y, true);
      assert(logFeats.size() == n);
      ublas_util::convertVec(logFeats, feats[y]);
      mass_y = exp(theta.w.innerProd(feats[y]));
      massTotal += mass_y;
      featsTotal += mass_y * feats[y];
    }
    
    // Update the function value.
    funcVal += log(massTotal) - theta.w.innerProd(feats[yi]);
    
    // Update the gradient wrt w.
    featsTotal /= massTotal;    
    subrange(gradFv, 0, n) += featsTotal - feats[yi];
  }
}

void LogLinearMultiELFV::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {

}
