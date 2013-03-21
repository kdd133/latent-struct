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

// Finkel and Manning set the domain-specific variance parameter to 1/10 that
// of the shared variance parameter. This is equivalent to setting the class-
// specific regularization coefficient to 10 times that of the shared
// regularization coefficient in our setting, which we adopt as the default.

RegularizerSoftTying::RegularizerSoftTying(double beta) : Regularizer(beta),
  _betaW(0), _betaSharedW(0), _betaU(0), _betaSharedU(0),
  _alphabet(0), _labels(0), _labelShared(-1) {
  assert(_beta > 0);
  setBeta(beta);
}

void RegularizerSoftTying::setBeta(double beta) {
  // If _beta == 0, we assume that processOptions was called, and do not
  // override those values. This is an ugly hack that's been put in place b/c
  // the main function in latent_struct.cpp always calls setBeta.
  if (_beta > 0) {
    cout << "RegularizerSoftTying: Setting default beta values relative to " <<
        "beta=" << beta << endl;
    _betaW = beta;
    _betaU = beta;
    // By default, we don't tie the classification parameters.
    _betaSharedW = 0; // 0.1 * beta;
    _betaSharedU = 0.1 * beta;
  }
}

int RegularizerSoftTying::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("w-beta", opt::value<double>(&_betaW),
        "regularization coefficient for class-specific parameters in w model")
    ("u-beta", opt::value<double>(&_betaU),
        "regularization coefficient for class-specific parameters in u model")
    ("shared-w-beta", opt::value<double>(&_betaSharedW),
        "regularization coefficient for shared parameters in w model")
    ("shared-u-beta", opt::value<double>(&_betaSharedU),
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
  
  // The setBeta method will only set the default relative values if _beta == 0.
  // In other words, if processOptions is called, and if at least one of the
  // beta values is set, then setBeta is disabled.
  if (vm.count("w-beta") || vm.count("u-beta") || vm.count("shared-w-beta") ||
      vm.count("shared-u-beta")) {
    _beta = 0; // disable setBeta
    
    // We also enforce the rule that if any of the four beta values is set via
    // the command line, then they must all be set.
    if (!vm.count("w-beta")) {
      cout << "RegularizerSoftTying: --w-beta was not specified" << endl;
      return 1;
    }
    if (!vm.count("u-beta")) {
      cout << "RegularizerSoftTying: --u-beta was not specified" << endl;
      return 1;
    }
    if (!vm.count("shared-w-beta")) {
      cout << "RegularizerSoftTying: --shared-w-beta was not specified" << endl;
      return 1;
    }
    if (!vm.count("shared-u-beta")) {
      cout << "RegularizerSoftTying: --shared-u-beta was not specified" << endl;
      return 1;
    }
  }
  
  return 0;
}

void RegularizerSoftTying::addRegularization(const Parameters& theta,
    double& fval, RealVec& grad) const {
  assert(_betaW >= 0 && _betaU >= 0 && _betaSharedW >= 0 && _betaSharedU >= 0);
  assert(_labelShared > 0);
  assert(_alphabet && _labels);
  assert(theta.getTotalDim() == grad.size());
  
  const Alphabet::DictType& featMap = _alphabet->getDict();
  Alphabet::DictType::const_iterator featIt;
  set<Label>::const_iterator labelIt;
  
  // If this is a w-u model, we need to offset the indices for the u portion of
  // the gradient.
  const int offsetU = theta.w.getDim();
  
  // Regularize the parameters for each class toward the shared parameters
  // using an L2 norm penalty.
  for (labelIt = _labels->begin(); labelIt != _labels->end(); ++labelIt) {
    const Label y = *labelIt;
    for (featIt = featMap.begin(); featIt != featMap.end(); ++featIt) {
      const string& f = featIt->first;
      const int fid = _alphabet->lookup(f, y, false);
      const int fid0 = _alphabet->lookup(f, _labelShared, false);
      assert(fid >= 0);
      const double diffW = theta.w[fid] - theta.w[fid0];
      fval += _betaW/2 * (diffW * diffW);
      grad(fid) += _betaW * diffW; // add beta*(w^y - w^0) to the gradient
      grad(fid0) -= _betaW * diffW; // update shared gradient (note sign)
      if (theta.hasU()) {
        const double diffU = theta.u[fid] - theta.u[fid0];
        fval += _betaU/2 * (diffU * diffU);
        grad(offsetU + fid) += _betaU * diffU;
        grad(offsetU + fid0) -= _betaU * diffU;
      }
    }
  }
  
  // Regularize the shared parameters toward zero using an L2 norm penalty.
  for (featIt = featMap.begin(); featIt != featMap.end(); ++featIt) {
    const int fid0 = _alphabet->lookup(featIt->first, _labelShared, false);
    fval += _betaSharedW/2 * (theta.w[fid0] * theta.w[fid0]);
    grad(fid0) += _betaSharedW * theta.w[fid0]; // add beta*w to gradient
    if (theta.hasU()) {
      fval += _betaSharedU/2 * (theta.u[fid0] * theta.u[fid0]);
      grad(offsetU + fid0) += _betaSharedU * theta.u[fid0];
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
  // the dummy label.
  const size_t n = alphabet.numFeaturesPerClass();
  _labelShared = ++maxLabel;
  alphabet.addLabel(_labelShared);
  theta.w.reAlloc(n * (labelSet.size() + 1));
  if (theta.hasU())
    theta.u.reAlloc(n * (labelSet.size() + 1));
}
