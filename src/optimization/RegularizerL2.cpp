/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Parameters.h"
#include "RegularizerL2.h"
#include "Ublas.h"
#include <assert.h>
#include <boost/program_options.hpp>

RegularizerL2::RegularizerL2(double beta) : Regularizer(beta), _betaW(0),
  _betaU(0) {
}

int RegularizerL2::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("beta-w", opt::value<double>(&_betaW),
        "regularization coefficient for w")
    ("beta-u", opt::value<double>(&_betaU),
        "regularization coefficient for u")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help")) {
    std::cout << options << std::endl;
    return 0;
  }
  return 0;
}

void RegularizerL2::addRegularization(const Parameters& theta, double& fval,
    RealVec& grad) const {
  const int d = theta.getTotalDim();
  assert(d == grad.size());
  
  // Note: The _beta value (from the parent class) will be ignored if either
  // _betaW > 0 or _betaU > 0. If _betaW == 0 and _betaU > 0, the w parameters
  // will not be regularized, and vice versa.
  if (_betaW > 0 || _betaU > 0) {
    if (_betaW > 0) {
      const int n = theta.w.getDim();
      fval += _betaW/2 * theta.w.squaredL2Norm();
      for (size_t i = 0; i < n; ++i)
        grad(i) += theta.w[i] * _betaW; // add beta*theta to gradient
    }
    if (_betaU > 0) {
      assert(theta.hasU());
      const int n = theta.u.getDim();
      fval += _betaU/2 * theta.u.squaredL2Norm();
      for (size_t i = 0; i < n; ++i)
        grad(i) += theta.u[i] * _betaU;
    }
  }
  else {
    assert(_beta > 0);
    fval += _beta/2 * theta.squaredL2Norm();
    for (size_t i = 0; i < d; ++i)
      grad(i) += theta[i] * _beta;
  }
}
