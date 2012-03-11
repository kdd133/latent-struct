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

double EmOptimizer::train(WeightVector& w, double tol) const {
  _objective.initLatentFeatureVectors(w);
  double valPrev = numeric_limits<double>::infinity();
  double valCur = valPrev;
  bool converged = false;  
  for (int iter = 0; iter < _maxIters; iter++) {
    boost::timer::auto_cpu_timer timer;
    _objective.setLatentFeatureVectors(w); // E-step (uses new W)
    valCur = _convexOpt->train(w, tol); // M-step (modifies W)
    
    if (valCur == numeric_limits<double>::infinity()) {
      cout << name() << ": Inner solver returned an infinite objective value. "
          << "Aborting via exit(1).\n";
      exit(1);
    }
    if (!_quiet)
      cout << name() << ": prev=" << valPrev << " current=" << valCur << endl; 
    if (valCur - valPrev > 0) {
      cout << name() << ": Objective value increased?! Terminating.\n";
      return valCur;
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
  
  if (!converged)
    cout << name() << ": Max iterations reached; objective value " << valCur
        << endl;
  return valCur;
}

void EmOptimizer::setBeta(double beta) {
  _beta = beta;
  _convexOpt->setBeta(beta);
}
