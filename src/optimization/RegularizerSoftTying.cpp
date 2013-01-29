/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Alphabet.h"
#include "Label.h"
#include "Parameters.h"
#include "RegularizerSoftTying.h"
#include "Ublas.h"
#include <assert.h>
#include <set>
#include <string>

using namespace std;

// TODO: Add the ability to specify distinct hyperparameter values for each
// class (i.e., beta_k).
// At least, we need to be able to specify different values for regularizing
// w and u.
RegularizerSoftTying::RegularizerSoftTying(double beta) : Regularizer(beta),
  _alphabet(0), _labels(0), _labelSharedW(-1), _labelSharedU(-1) {
}

void RegularizerSoftTying::addRegularization(const Parameters& theta,
    double& fval, RealVec& grad) const {
  assert(_beta > 0.0);
  assert(_labelSharedW > 0);
  assert(_labelSharedU > 0 || !theta.hasU());
  assert(_alphabet && _labels);
  
  const Alphabet::DictType& featMap = _alphabet->getDict();
  Alphabet::DictType::const_iterator featIt;
  set<Label>::const_iterator labelIt;
  
  // Regularize the parameters for each class toward the shared parameters
  // using an L2 norm penalty.
  for (labelIt = _labels->begin(); labelIt != _labels->end(); ++labelIt) {
    const Label y = *labelIt;
    for (featIt = featMap.begin(); featIt != featMap.end(); ++featIt) {
      const string& f = featIt->first;
      const int fid = _alphabet->lookup(f, y, false);
      int fid0 = _alphabet->lookup(f, _labelSharedW, false);
      assert(fid >= 0);
      const double diffW = theta.w[fid] - theta.w[fid0];
      fval += _beta/2 * (diffW * diffW);
      grad(fid) += _beta * diffW; // add beta*(w^y - w^0) to the gradient
      grad(fid0) -= _beta * diffW; // update the shared gradient (note the sign)
      if (theta.hasU()) {
        fid0 = _alphabet->lookup(f, _labelSharedU, false);
        const double diffU = theta.u[fid] - theta.u[fid0];
        fval += _beta/2 * (diffU * diffU);
        grad(fid) += _beta * diffU;
        grad(fid0) -= _beta * diffU;
      }
    }
  }
  
  // Regularize the shared parameters toward zero using an L2 norm penalty.
  const int d = theta.getTotalDim();
  assert(d == grad.size());
  for (featIt = featMap.begin(); featIt != featMap.end(); ++featIt) {
    int fid0 = _alphabet->lookup(featIt->first, _labelSharedW, false);
    fval += _beta/2 * (theta.w[fid0] * theta.w[fid0]);
    grad(fid0) += _beta * theta.w[fid0]; // add beta*w to gradient
    if (theta.hasU()) {
      fid0 = _alphabet->lookup(featIt->first, _labelSharedU, false);
      fval += theta.u[fid0] * theta.u[fid0];
      grad(fid0) += _beta * theta.u[fid0];
    }
  }
}

void RegularizerSoftTying::setupParameters(Parameters& theta,
    Alphabet& alphabet, const set<Label>& labelSet) {
  // Deteremine the maximum label value.
  Label maxLabel = -1;
  set<Label>::const_iterator it;
  for (it = labelSet.begin(); it != labelSet.end(); ++it)
    if (*it > maxLabel)
      maxLabel = *it;
  assert(maxLabel > 0);
  
  // Store a pointer to the alphabet and label set, as we will need these in
  // the addRegularization method.
  _alphabet = &alphabet;
  _labels = &labelSet;
  
  // Add a dummy label for the shared parameters to the alphabet. Note that we
  // do not add the label to the label set, so that the learner is unaware of
  // the dummy label. Repeat for the u model (latent variable imputation model)
  // if present.
  const size_t n = alphabet.size();
  _labelSharedW = ++maxLabel;
  alphabet.addLabel(_labelSharedW);
  theta.w.reAlloc(n * (labelSet.size() + 1));
  if (theta.hasU()) {
    _labelSharedU = ++maxLabel;
    alphabet.addLabel(_labelSharedU);
    theta.u.reAlloc(n * (labelSet.size() + 1));
  }
}
