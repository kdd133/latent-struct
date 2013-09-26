/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "EmOptimizer.h"
#include "Model.h"
#include "Optimizer.h"
#include "Parameters.h"
#include "Regularizer.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include "ValidationSetHandler.h"
#include <assert.h>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>

using namespace boost;
using namespace std;

EmOptimizer::EmOptimizer(shared_ptr<TrainingObjective> objective,
                         shared_ptr<Regularizer> regularizer,
                         shared_ptr<Optimizer> opt) :
    Optimizer(objective, regularizer), _convexOpt(opt), _maxIters(20),
      _abortOnConsecMaxIters(false), _quiet(false), _ignoreIncrease(false) {
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
    ("em-ignore-increase", opt::bool_switch(&_ignoreIncrease),
        "continue iterating if the objective value is seen to increase")
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

Optimizer::status EmOptimizer::train(Parameters& w, double& valCur,
    double tol) const {
  _objective->initLatentFeatureVectors(w);
  double valPrev = numeric_limits<double>::infinity();
  bool converged = false;  
  bool innerMaxItersPrev = false; // True if inner solver reached max iterations
                                  // on the previous EM iteration.
  for (int iter = 0; iter < _maxIters; iter++) {
    boost::timer::cpu_timer timer;
    
    // E-step (uses new W)
    _objective->setLatentFeatureVectors(w);
    
    // M-step (modifies W)
    const Optimizer::status status = _convexOpt->train(w, valCur, tol);
    
    if (status == Optimizer::FAILURE) {
      cout << name() << " iter = " << iter <<
          ": Inner solver reported a failure. Terminating.\n";
      getBestOnValidation(w, valCur);
      return Optimizer::FAILURE;
    }
    
    if (!_quiet) {
      cout << name() << " iter = " << iter << ": prev=" << valPrev <<
          " current=" << valCur;
      cout << " timer:" << timer.format();
    }
    if (_validationSetHandler)
      _validationSetHandler->evaluate(w, iter);

    if (status == Optimizer::BACKWARD_PROGRESS) {
      cout << name() << " iter = " << iter <<
          ": Inner solver reported backward progress. Terminating.\n";
      getBestOnValidation(w, valCur);
      return status;
    }
    
    if (_abortOnConsecMaxIters) {
      if (status == Optimizer::MAX_ITERS_CONVEX) {
        if (innerMaxItersPrev) {
          cout << name() << " iter = " << iter <<
              ": Inner solver reached max iterations on two " <<
              "consecutive EM iterations. Terminating\n";
          getBestOnValidation(w, valCur);
          return Optimizer::MAX_ITERS_CONVEX;
        }
        innerMaxItersPrev = true;
      }
      else {
        innerMaxItersPrev = false;
      }
    }
    
    if (valCur - valPrev > 0) {
      if (_ignoreIncrease) {
        cout << name() << " iter = " << iter <<
            ": [Warning] Objective value increased.\n";
        valPrev = valCur;
        continue; // i.e., skip the convergence check
      }
      cout << name() << " iter = " << iter <<
          ": Objective value increased?! Terminating.\n";
      getBestOnValidation(w, valCur);
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
    getBestOnValidation(w, valCur);
    return Optimizer::MAX_ITERS_ALTERNATING;
  }
  getBestOnValidation(w, valCur);
  return Optimizer::CONVERGED;
}

void EmOptimizer::getBestOnValidation(Parameters& w, double& score) const {
  // If we evaluated on a validation set, return the best parameters we obtained
  // according to the chosen performance metric. Otherwise, return the current
  // parameters w.
  if (_validationSetHandler) {
    const Parameters& wBest = _validationSetHandler->getBestParams();
    assert(wBest.getDimTotal() > 0);
    assert(wBest.getDimTotal() == w.getDimTotal());
    w.setParams(wBest);
    score = _validationSetHandler->getBestScore();
    if (!_quiet) {
      cout << name() << ": Highest performance achieved on validation set was "
          << _validationSetHandler->getBestScore() << " "
          << _validationSetHandler->getPerfMeasure() << endl;
    }
  }
}
