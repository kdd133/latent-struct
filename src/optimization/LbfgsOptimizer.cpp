/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "FeatureVector.h"
#include "LbfgsOptimizer.h"
#include "Model.h"
#include "RealWeight.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <iostream>
#include <lbfgs.h>
using namespace std;


LbfgsOptimizer::LbfgsOptimizer(TrainingObjective& objective) :
    Optimizer(objective), _restarts(3), _quiet(false) {
  lbfgs_parameter_init(&_params);
}

int LbfgsOptimizer::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("beta", opt::value<double>(&_beta)->default_value(0.5),
        "the L2 regularization constant, i.e., (beta/2)*||w||^2")
    ("epsilon", opt::value<lbfgsfloatval_t >(&_params.epsilon)
        ->default_value(1e-4), "value used when testing for convergence")
    ("m", opt::value<int>(&_params.m)->default_value(20),
        "number of vectors used to compute the approximate the inverse Hessian")
    ("max-iters", opt::value<int>(&_params.max_iterations)->default_value(250),
        "maximum number of iterations")
    ("max-linesearch",
        opt::value<int>(&_params.max_linesearch)->default_value(5),
        "maximum number of line search iterations")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
    ("restarts", opt::value<int>(&_restarts)->default_value(3),
        "number of times to restart Lbfgs when it thinks it has converged")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help"))
    cout << options << endl;
  return 0;
}

lbfgsfloatval_t LbfgsOptimizer::evaluate(void* instance, const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g, const int d, const lbfgsfloatval_t step) {
  assert(instance != 0);
  assert(g != 0);
  assert(x != 0);
  
  boost::timer::auto_cpu_timer timer;
  
  const LbfgsInstance* inst = (LbfgsInstance*)instance;
  TrainingObjective& obj = *inst->obj;
  WeightVector& w = *inst->w;
  const double beta = inst->beta;
  double fval = 0;
  
  // Set our model to the current point x.
  w.setWeights(x, d);
  
  // Compute the gradient at the given w.
  // Note: The above setting of w updated the model used here by obj.
  FeatureVector<RealWeight> gradFv(d);
  obj.valueAndGradient(w, fval, gradFv);
  Utility::addRegularizationL2(w, beta, fval, gradFv);
  
  // Copy the new gradient back into g, for return to lbfgs.
  for (int i = 0; i < d; i++)
    g[i] = gradFv.getValueAtLocation(i);
  
  return fval;
}

int LbfgsOptimizer::progress(void* instance, const lbfgsfloatval_t* x,
    const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls) {
    
  cout << "iter = " << k << ", ";
  cout << "fx = " << fx << ", ";
  cout << "ls = " << ls << ", ";
  cout << "xnorm = " << xnorm << ", ";
  cout << "gnorm = " << gnorm << ", ";
  cout << "step = " << step
      << endl;
      
  return 0;
}

double LbfgsOptimizer::train(WeightVector& w) const {
  boost::timer::auto_cpu_timer timer;
  const int d = w.getDim();
  assert(d > 0);
  
  LbfgsInstance inst = { &_objective, &w, _beta };
  lbfgsfloatval_t* x = lbfgs_malloc(d);
  for (int i = 0; i < d; i++)
    x[i] = w.getWeight(i); // set the starting point to be w
  lbfgsfloatval_t objVal = 0.0;
  int ret = -1;
  
  for (int t = 0; t < _restarts; t++) {
    lbfgs_parameter_t params = _params; // make a copy, since train() is const
    ret = lbfgs(d, x, &objVal, evaluate, _quiet ? 0 : progress, &inst, &params);
    bool terminate = false;
    switch (ret) {
      cout << name() << ": ";
      case LBFGSERR_ROUNDING_ERROR:
      case LBFGSERR_MINIMUMSTEP:
      case LBFGSERR_MAXIMUMSTEP:
      case LBFGSERR_INCREASEGRADIENT:
      case LBFGSERR_INVALIDPARAMETERS:
        cout << "Caught non-convex related error (code " << ret <<
          "). Performing restart.\n";
        break;
      case LBFGS_CONVERGENCE:
        cout << "Convergence detected. Performing restart.\n";
        break;
      case LBFGSERR_MAXIMUMLINESEARCH:
        cout << "Reached max number of line search iterations. Terminating\n";
        terminate = true;
        break;
      case LBFGSERR_MAXIMUMITERATION:
        cout << "Reached maximum number of iterations. Terminating.\n";
        terminate = true;
        break;
      case LBFGS_ALREADY_MINIMIZED:
        cout << "Function appears to be already minimized. Terminating.\n";
        terminate = true;
        break;
      default:
        cout << "lbfgs() returned code " << ret << ". Terminating.\n";
        terminate = true;
        break;
    }
    if (terminate)
      break;
  }
  cout << name() << ": Optimization terminated with objective value " << objVal
      << endl; 
  w.setWeights(x, d); // copy the final point into w
  
  lbfgs_free(x);
  return objVal;
}
