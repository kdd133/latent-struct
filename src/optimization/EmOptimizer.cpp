/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "EmOptimizer.h"
#include "FeatureVector.h"
#include "Model.h"
#include "Optimizer.h"
#include "RealWeight.h"
#include "TrainingObjective.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

EmOptimizer::EmOptimizer(TrainingObjective& objective,
    shared_ptr<Optimizer> opt) :
    Optimizer(objective, 1e-4), _convexOpt(opt), _maxIters(20),
      _abortOnConsecMaxIters(false), _quiet(false) {
}

int EmOptimizer::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("em-abort-on-consec-max-iters", opt::bool_switch(&_abortOnConsecMaxIters),
        "abort if the inner solver exceeds its maximum number of iterations \
on two consecutive EM iterations")
    ("em-max-iters", opt::value<int>(&_maxIters)->default_value(20),
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
  bool innerMaxItersPrev = false; // True if inner solver reached max iterations
                                  // on the previous EM iteration.
  for (int iter = 0; iter < _maxIters; iter++) {
//    boost::timer::auto_cpu_timer timer; // Uncomment to print timing info.
    
    // E-step (uses new W)
    _objective.setLatentFeatureVectors(w);
    
    // M-step (modifies W)
    const Optimizer::status status = _convexOpt->train(w, valCur, tol);
    
    if (status == Optimizer::FAILURE) {
      cout << name() << " iter = " << iter <<
          ": Inner solver reported a failure. Terminating.\n";
      return Optimizer::FAILURE;
    }
    
    if (!_quiet)
      cout << name() << " iter = " << iter << ": prev=" << valPrev <<
          " current=" << valCur << endl;
      
    if (status == Optimizer::BACKWARD_PROGRESS) {
      cout << name() << " iter = " << iter <<
          ": Inner solver reported backward progress. Terminating.\n";
      return status;
    }
    
    if (_abortOnConsecMaxIters) {
      if (status == Optimizer::MAX_ITERS_CONVEX) {
        if (innerMaxItersPrev) {
          cout << name() << " iter = " << iter <<
              ": Inner solver reached max iterations on two " <<
              "consecutive EM iterations. Terminating\n";
          return Optimizer::MAX_ITERS_CONVEX;
        }
        innerMaxItersPrev = true;
      }
      else {
        innerMaxItersPrev = false;
      }
    }
    
    if (valCur - valPrev > 0) {
      cout << name() << " iter = " << iter <<
          ": Objective value increased?! Terminating.\n";
      return Optimizer::BACKWARD_PROGRESS;
    }
    
    if (valPrev - valCur < tol) {
      if (!_quiet)
        cout << name() << " iter = " << iter <<
            ": Convergence detected; objective value " << valCur << endl;
      converged = true;
      break;
    }
    
    valPrev = valCur;
  }
  
  if (!converged) {
    cout << name() << ": Max iterations reached; objective value " << valCur
        << endl;
    return Optimizer::MAX_ITERS_ALTERNATING;
  }
  return Optimizer::CONVERGED;
}

void EmOptimizer::setBeta(double beta) {
  _beta = beta;
  _convexOpt->setBeta(beta);
}
