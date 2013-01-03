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
#include "LogLinearMultiUW.h"
#include "LogWeight.h"
#include "Model.h"
#include "Parameters.h"
#include "Ublas.h"
#include <assert.h>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/shared_array.hpp>
#include <vector>

void LogLinearMultiUW::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, RealVec& gradFv) {
  
  const int n = theta.w.getDim();    // i.e., the number of features
  const int d = theta.getTotalDim(); // i.e., the length of the [w u] vector
  assert(n > 0 && n == theta.u.getDim());
  assert(d == gradFv.size());
  
  LogVec logFeatsW(n);
  LogVec logFeatsU_yi;  // call to expectedFeatureCooccurrences will allocate
  LogMat logCoocU_yi;   // call to expectedFeatureCooccurrences will allocate
  
  LogVec logFeats;      // call to expectedFeatures will allocate
  RealVec feats(n);
  RealVec gradU(n);
  RealMat covU_yi(n, n);
  
  funcVal = 0;
  gradFv.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    // Compute the mass and the expected features wrt w.
    LogWeight massW(0);
    logFeatsW.clear();
    for (Label y = 0; y < k; y++) {      
      // Note: The last argument is false b/c we want unnormalized features.
      massW += model.expectedFeatures(theta.w, logFeats, xi, y, false);
      assert(logFeats.size() == n);
      logFeatsW += logFeats;
    }    
    // Normalize the expected features under w.
    logFeatsW /= massW;
    
    // Compute the mass, expected features, and feature co-occurrences wrt u.
    // Note: logFeatsU_yi and logCoocU_yi will be normalized after this call.
    LogWeight massU_yi = model.expectedFeatureCooccurrences(theta.u,
        logCoocU_yi, logFeatsU_yi, xi, yi);
        
    // Compute the matrix of feature covariances.
    ublas_util::convertVec(logFeatsU_yi, feats);
    ublas_util::convertMat(logCoocU_yi, covU_yi);
    covU_yi = covU_yi - outer_prod(feats, feats);
    
    // Compute u-w and store the result in feats.
    ublas_util::subtractWeightVectors(theta.u, theta.w, feats);
    
    // Compute covU_yi' * (u-w) and store the result in gradU.
    axpy_prod(feats, covU_yi, gradU, true);
    
    // Update the gradient wrt w (first, exponentiate features).
    subrange(gradFv, 0, n) += ublas_util::convertVec(logFeatsW, feats);
    subrange(gradFv, 0, n) -= ublas_util::convertVec(logFeatsU_yi, feats);
    
    // Update the gradient wrt u.
    subrange(gradFv, n, d) += gradU;
    
    // Update the function value.
    // Note: We work in log space here, which is why we don't exponentiate.
    funcVal += theta.u.innerProd(logFeatsU_yi) - theta.w.innerProd(logFeatsU_yi); 
    funcVal += massW;
    funcVal -= massU_yi;
  }
}

void LogLinearMultiUW::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  assert(0); // This is the w-only version. u-w not implemented yet.
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      const double yScore = model.totalMass(theta.w, x, y);
      scores.setScore(id, y, yScore);
    }
  }
}
