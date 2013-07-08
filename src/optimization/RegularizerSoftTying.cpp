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
#include "Utility.h"
#include <assert.h>
#include <boost/program_options.hpp>
#include <set>
#include <string>

using namespace std;

RegularizerSoftTying::RegularizerSoftTying(double beta) : Regularizer(beta),
  _betaW(0), _betaSharedW(0), _betaU(0), _betaSharedU(0),
  _alphabet(0), _labels(0) {
  assert(_beta > 0);
  setBeta(beta);
}

// Finkel and Manning set the domain-specific variance parameter to 1/10 that
// of the shared variance parameter. This is equivalent to setting the class-
// specific regularization coefficient to 10 times that of the shared
// regularization coefficient in our setting, which we adopt as the default.
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
    double& fval, SparseRealVec& grad) const {
  assert(_betaW >= 0 && _betaU >= 0 && _betaSharedW >= 0 && _betaSharedU >= 0);
  assert(_alphabet && _labels);
  assert(theta.getDimTotal() == grad.size());
  assert(theta.shared_w.getDim() > 0);
  assert(!theta.hasU() || theta.shared_u.getDim() == theta.shared_w.getDim());
  
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
      const int fid0 = _alphabet->getFeatureIndex(f);
      assert(fid >= 0 && fid0 >= 0);
      const double diffW = theta.w[fid] - theta.shared_w[fid0];
      fval += _betaW/2 * (diffW * diffW);
      // add beta*(w^y - w^0) to the gradient
      grad(theta.indexW() + fid) += _betaW * diffW;
      // update shared gradient (note the sign flip)
      grad(theta.indexSharedW() + fid0) -= _betaW * diffW;
      if (theta.hasU()) {
        const double diffU = theta.u[fid] - theta.shared_u[fid0];
        fval += _betaU/2 * (diffU * diffU);
        grad(theta.indexU() + fid) += _betaU * diffU;
        grad(theta.indexSharedU() + fid0) -= _betaU * diffU;
      }
    }
  }
  
  // Regularize the shared parameters toward zero using an L2 norm penalty.
  for (featIt = featMap.begin(); featIt != featMap.end(); ++featIt) {
    const int fid0 = _alphabet->getFeatureIndex(featIt->first);
    assert(fid0 >= 0);
    fval += _betaSharedW/2 * (theta.shared_w[fid0] * theta.shared_w[fid0]);
    // add beta*w to gradient
    grad(theta.indexSharedW() + fid0) += _betaSharedW * theta.shared_w[fid0];
    if (theta.hasU()) {
      fval += _betaSharedU/2 * (theta.shared_u[fid0] * theta.shared_u[fid0]);
      grad(theta.indexSharedU() + fid0) += _betaSharedU * theta.shared_u[fid0];
    }
  }
}

void RegularizerSoftTying::setupParameters(Parameters& theta,
    Alphabet& alphabet, const set<Label>& labelSet, int seed) {
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
  
  const size_t n = alphabet.numFeaturesPerClass();
  theta.shared_w.reAlloc(n);
  // Initialize theta.shared_w to random values, but only if _betaSharedW has
  // been set to a positive value. Otherwise, the random initial values will
  // influence the fval even if _betaSharedW is zero.
  if (_betaSharedW > 0) {
    boost::shared_array<double> w0 = Utility::generateGaussianSamples(n, 0,
      0.01, seed);
    theta.shared_w.setWeights(w0.get(), n);
  }
  if (theta.hasU()) {
    theta.shared_u.reAlloc(n);
    if (_betaSharedU > 0) {
      // Note: We increment the seed to avoid symmetry in the parameters.
      boost::shared_array<double> u0 = Utility::generateGaussianSamples(n, 0,
          0.01, seed + 1);
      theta.shared_u.setWeights(u0.get(), n);
    }
  }
}
