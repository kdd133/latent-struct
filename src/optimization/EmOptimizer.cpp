/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <assert.h>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
#include "FeatureVector.h"
#include "EmOptimizer.h"
#include "Model.h"
#include "Optimizer.h"
#include "RealWeight.h"
#include "TrainingObjective.h"
#include "WeightVector.h"

EmOptimizer::EmOptimizer(TrainingObjective& objective,
    shared_ptr<Optimizer> opt) :
    Optimizer(objective, 1e-4), _convexOpt(opt), _maxIters(10) {
}

int EmOptimizer::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("em-max-iters", opt::value<int>(&_maxIters)->default_value(10),
        "maximum number of iterations")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
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

Optimizer::status EmOptimizer::train(WeightVector& w, double& valCur,
    double tol) const {
  _objective.initLatentFeatureVectors(w);
  double valPrev = numeric_limits<double>::infinity();
  bool converged = false;  
  for (int iter = 0; iter < _maxIters; iter++) {
    boost::timer::auto_cpu_timer timer;
    
    // E-step (uses new W)
    _objective.setLatentFeatureVectors(w);
    
    // M-step (modifies W)
    const Optimizer::status status = _convexOpt->train(w, valCur, tol);
    
    if (status == Optimizer::FAILURE) {
      cout << name() << ": Inner solver reported a failure. Terminating.\n";
      return Optimizer::FAILURE;
    }
    
    if (!_quiet)
      cout << name() << ": prev=" << valPrev << " current=" << valCur << endl;
      
    if (status == Optimizer::BACKWARD_PROGRESS) {
      cout << name() << ": Inner solver reported backward progress. " <<
          "Terminating.\n";
      return status;
    }
    if (status == Optimizer::MAX_ITERS) {
      cout << name() << ": Inner solver reached max iterations. Terminating\n";
      return status;
    }
    if (valCur - valPrev > 0) {
      cout << name() << ": Objective value increased?! Terminating.\n";
      return Optimizer::BACKWARD_PROGRESS;
    }
    if (valPrev - valCur < tol) {
      if (!_quiet)
        cout << name() << ": Convergence detected; objective value " << valCur
          << endl;
      converged = true;
      break;
    }
    valPrev = valCur;
  }
  
  if (!converged) {
    cout << name() << ": Max iterations reached; objective value " << valCur
        << endl;
    return Optimizer::MAX_ITERS;
  }
  return Optimizer::CONVERGED;
}

void EmOptimizer::setBeta(double beta) {
  _beta = beta;
  _convexOpt->setBeta(beta);
}
