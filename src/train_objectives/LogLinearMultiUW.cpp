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
#include <vector>

//#define USE_SLOW_GRAD


void LogLinearMultiUW::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, RealVec& gradFv) {
  
  const int n = theta.w.getDim();    // i.e., the number of features
  const int d = theta.getTotalDim(); // i.e., the length of the [w u] vector
  assert(theta.hasU());
  assert(n > 0 && n == theta.u.getDim());
  assert(d == gradFv.size());
  
  SparseLogVec logFeatsW(n);
  SparseLogVec logFeatsU_yi;  // call to expectedFeatureCooccurrences will allocate
  
  SparseLogVec logFeats;      // call to expectedFeatures will allocate
  SparseRealVec feats(n);
  SparseRealVec gradU(n);
  RealVec uMinusW(n);
  AccumRealMat covU_yi(n, n);
  
  // Compute u-w.
  ublas_util::subtractWeightVectors(theta.u, theta.w, uMinusW);
  
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
    LogWeight massU_yi;
    const AccumLogMat* logCoocU_yi = model.expectedFeatureCooccurrences(theta.u,
        massU_yi, logFeatsU_yi, xi, yi, true);
    assert(logCoocU_yi);
        
    // Compute the matrix of feature covariances.
    ublas_util::exponentiate(logFeatsU_yi, feats);
#ifdef USE_SLOW_GRAD
    // Fill in the upper portion of the logCoocU_yi matrix.
    AccumLogMat logCoocU_yi_full = *logCoocU_yi;
    AccumLogMat::const_iterator1 it1;
    AccumLogMat::const_iterator2 it2;
    for (it1 = logCoocU_yi_full.begin1(); it1 != logCoocU_yi_full.end1(); ++it1)
      for (it2 = it1.begin(); it2 != it1.end() && it2.index2() < it2.index1();
          ++it2)
        logCoocU_yi_full(it2.index2(), it2.index1()) = *it2;
        
    ublas_util::exponentiate(logCoocU_yi_full, covU_yi);
    covU_yi -= outer_prod(feats, feats);
    
    // Compute covU_yi' * (u-w) and store the result in gradU.
    axpy_prod(uMinusW, covU_yi, gradU, true);
#else   
    ublas_util::computeLowerCovarianceMatrix(*logCoocU_yi, feats, covU_yi);
    ublas_util::matrixVectorMultLowerSymmetric(covU_yi, uMinusW, gradU);
#endif
    
    // Update the gradient wrt w (first, exponentiate features).
    subrange(gradFv, 0, n) += ublas_util::exponentiate(logFeatsW, feats);
    subrange(gradFv, 0, n) -= ublas_util::exponentiate(logFeatsU_yi, feats);
    
    // Update the gradient wrt u.
    subrange(gradFv, n, d) += gradU;
    
    // Update the function value.
    // Note: We work in log space here, which is why we don't exponentiate.
    funcVal += inner_prod(uMinusW, feats); // add (u-w)*featsU_yi
    funcVal += massW;
    funcVal -= massU_yi;
    
//    if (xi.getId() > 500)
//      break; // for debugging purposes
  }
}

void LogLinearMultiUW::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const int n = theta.w.getDim();
  RealVec wMinusU(n);
  ublas_util::subtractWeightVectors(theta.w, theta.u, wMinusU);
  SparseLogVec logFeatsU;
  SparseRealVec featsU(n);
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      const double massU = model.expectedFeatures(theta.u, logFeatsU, x, y,
          true);
      ublas_util::exponentiate(logFeatsU, featsU);
      const double yScore = inner_prod(wMinusU, featsU) + massU;
      scores.setScore(id, y, yScore);
    }
  }
}
