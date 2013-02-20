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
  
  std::vector<SparseRealVec> phiBar(k, LogVec(n));
  std::vector<SparseRealMat> cov(k, RealMat(n, n));
  SparseLogVec logFeats(n);
  SparseRealVec phiBar_sumY(n);
  SparseRealMat cooc(n, n);
  SparseRealMat covTotal(n, n);
  SparseRealVec gradU(n);
  AccumLogMat logCooc(n, n);
  double massTotal;
  
  // Copy theta.w into RealVec w.
  RealVec w(n);
  for (size_t i = 0; i < n; ++i)
    w(i) = theta.w[i];
  
  funcVal = 0;
  gradFv.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    massTotal = 0;
    phiBar_sumY.clear();
    covTotal.clear();
    
    for (Label y = 0; y < k; y++) {      
      // Get the (normalized) log expected features and coocurrences. 
      model.expectedFeatureCooccurrences(theta.u, &logCooc, &logFeats, xi, y);
      assert(logFeats.size() == n);
      
      // Exponentiate the log values.
      ublas_util::exponentiate(logCooc, cooc);
      ublas_util::exponentiate(logFeats, phiBar[y]);
      
      const double mass_y = exp(theta.w.innerProd(phiBar[y]));
      massTotal += mass_y;
      phiBar_sumY += mass_y * phiBar[y];
      cov[y] = cooc - outer_prod(phiBar[y], phiBar[y]);
      covTotal += mass_y * cov[y];
    }
    
    // Normalize the counts.
    phiBar_sumY /= massTotal;
    covTotal /= massTotal;

    // Compute w'*covTotal and store the result in gradU.
    axpy_prod(w, covTotal, gradU, true);  // true --> do gradU.clear() first
    axpy_prod(-w, cov[yi], gradU, false); // false --> add -w'*cov[yi] to gradU

    // Update the function value.
    funcVal += log(massTotal) - theta.w.innerProd(phiBar[yi]);
    
    // Update the gradient wrt w.
    subrange(gradFv, 0, n) += phiBar_sumY - phiBar[yi];    
    
    // Update the gradient wrt u.
    subrange(gradFv, n, d) += gradU;
  }
}

void LogLinearMultiELFV::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  SparseLogVec logFeats(theta.w.getDim());
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      model.expectedFeatures(theta.u, &logFeats, x, y, true);
      const double yScore = theta.w.innerProd(logFeats);
      scores.setScore(id, y, yScore);
    }
  }
}
