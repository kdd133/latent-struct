/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Label.h"
#include "Parameters.h"
#include "RegularizerSoftTying.h"
#include "Ublas.h"
#include <assert.h>
#include <set>

using std::set;

// TODO: Add the ability to specify distinct hyperparameter values for each
// class (i.e., beta_k).
RegularizerSoftTying::RegularizerSoftTying(double beta) : Regularizer(beta),
  _labelSharedW(-1), _labelSharedU(-1) {
}

void RegularizerSoftTying::addRegularization(const Parameters& theta,
    double& fval, RealVec& grad) const {
  assert(_labelSharedW > 0);
  assert(_labelSharedU > 0 || !theta.hasU());
  assert(0); // FIXME: Not implemented yet.
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
  
  // Add a dummy label for the shared parameters to the alphabet. Note that we
  // do not add the label to the label set, so that the learner is unaware of
  // the dummy label. Repeat for the u model (latent variable imputation model)
  // if present.
  _labelSharedW = ++maxLabel;
  alphabet.addLabel(_labelSharedW);
  if (theta.hasU()) {
    _labelSharedU = ++maxLabel;
    alphabet.addLabel(_labelSharedU);
  }
}
