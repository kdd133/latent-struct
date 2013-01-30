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
#include <boost/program_options.hpp>
#include <set>
#include <string>

using namespace std;

RegularizerSoftTying::RegularizerSoftTying(double beta) : Regularizer(beta),
  _betaW(beta), _betaSharedW(10*beta), _betaU(beta), _betaSharedU(10*beta),
  _alphabet(0), _labels(0), _labelSharedW(-1), _labelSharedU(-1) {
}

int RegularizerSoftTying::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("beta-w", opt::value<double>(&_betaW),
        "regularization coefficient for class-specific parameters in w model")
    ("beta-u", opt::value<double>(&_betaU),
        "regularization coefficient for class-specific parameters in u model")
    ("beta-shared-w", opt::value<double>(&_betaSharedW),
        "regularization coefficient for shared parameters in w model")
    ("beta-shared-u", opt::value<double>(&_betaSharedU),
        "regularization coefficient for shared parameters in u model")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help")) {
    cout << options << endl;
    return 0;
  }
  return 0;
}

void RegularizerSoftTying::addRegularization(const Parameters& theta,
    double& fval, RealVec& grad) const {
  assert(_betaW > 0 && _betaU > 0 && _betaSharedW > 0 && _betaSharedU > 0);
  assert(_labelSharedW > 0);
  assert(_labelSharedU > 0 || !theta.hasU());
  assert(_alphabet && _labels);
  assert(theta.getTotalDim() == grad.size());
  
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
      fval += _betaW/2 * (diffW * diffW);
      grad(fid) += _betaW * diffW; // add beta*(w^y - w^0) to the gradient
      grad(fid0) -= _betaSharedW * diffW; // update shared gradient (note sign)
      if (theta.hasU()) {
        fid0 = _alphabet->lookup(f, _labelSharedU, false);
        const double diffU = theta.u[fid] - theta.u[fid0];
        fval += _betaU/2 * (diffU * diffU);
        grad(fid) += _betaU * diffU;
        grad(fid0) -= _betaSharedU * diffU;
      }
    }
  }
  
  // Regularize the shared parameters toward zero using an L2 norm penalty.
  for (featIt = featMap.begin(); featIt != featMap.end(); ++featIt) {
    int fid0 = _alphabet->lookup(featIt->first, _labelSharedW, false);
    fval += _betaSharedW/2 * (theta.w[fid0] * theta.w[fid0]);
    grad(fid0) += _betaSharedW * theta.w[fid0]; // add beta*w to gradient
    if (theta.hasU()) {
      fid0 = _alphabet->lookup(featIt->first, _labelSharedU, false);
      fval += _betaSharedU/2 * (theta.u[fid0] * theta.u[fid0]);
      grad(fid0) += _betaSharedU * theta.u[fid0];
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
  const size_t n = alphabet.numFeaturesPerClass();
  _labelSharedW = ++maxLabel;
  alphabet.addLabel(_labelSharedW);
  theta.w.reAlloc(n * (labelSet.size() + 1));
  if (theta.hasU()) {
    _labelSharedU = ++maxLabel;
    alphabet.addLabel(_labelSharedU);
    theta.u.reAlloc(n * (labelSet.size() + 1));
  }
}
