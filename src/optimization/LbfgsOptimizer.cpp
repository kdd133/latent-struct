/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "LbfgsOptimizer.h"
#include "Model.h"
#include "Optimizer.h"
#include "Parameters.h"
#include "Regularizer.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include "Utility.h"
#include "ValidationSetHandler.h"
#include <assert.h>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <iostream>
#include <lbfgs.h>

using namespace boost;
using namespace std;

LbfgsOptimizer::LbfgsOptimizer(shared_ptr<TrainingObjective> objective,
                               shared_ptr<Regularizer> regularizer) :
    Optimizer(objective, regularizer), _restarts(3), _quiet(false) {
  lbfgs_parameter_init(&_params);
}

int LbfgsOptimizer::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
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
  
  boost::timer::cpu_timer timer;
  
  const LbfgsInstance* inst = (LbfgsInstance*)instance;
  TrainingObjective* obj = inst->obj.get();
  const Regularizer* reg = inst->reg.get();
  Parameters& theta = *inst->theta;
  double fval = 0;
  
  // Set our model to the current point x.
  theta.setWeights(x, d);
  
  // Compute the gradient at the given theta.
  // Note: The above setting of theta updated the model used here by obj.
  SparseRealVec gradFv(d);
  obj->valueAndGradient(theta, fval, gradFv);
  reg->addRegularization(theta, fval, gradFv);
  
  // Copy the new gradient back into g, for return to lbfgs.
  for (int i = 0; i < d; i++)
    g[i] = gradFv(i);
  
  if (!inst->quiet)
    cout << timer.format();

  return fval;
}

int LbfgsOptimizer::progress(void* instance, const lbfgsfloatval_t* x,
    const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls) {
    
  cout << "LbfgsOptimizer: ";
  cout << "iter = " << k << ", ";
  cout << "fx = " << fx << ", ";
  cout << "ls = " << ls << ", ";
  cout << "xnorm = " << xnorm << ", ";
  cout << "gnorm = " << gnorm << ", ";
  cout << "step = " << step
      << endl;
      
  // Evaluate on the validation set if one is present.
  LbfgsInstance* inst = (LbfgsInstance*)instance;
  if (inst->vsh) {
    Parameters& theta = *inst->theta;
    inst->vsh->evaluate(theta, k);
  }

  return 0;
}

Optimizer::status LbfgsOptimizer::train(Parameters& theta, double& fval,
    double tol) const {
  boost::timer::cpu_timer timer;
  const int d = theta.getDimTotal();
  assert(d > 0);
  
  if (_validationSetHandler)
    _validationSetHandler->clearBest();
  
  LbfgsInstance inst;
  inst.obj = _objective;
  inst.reg = _regularizer;
  inst.theta = &theta;
  inst.quiet = _quiet;
  inst.vsh = _validationSetHandler;
  
  lbfgsfloatval_t* x = lbfgs_malloc(d);
  for (int i = 0; i < d; i++)
    x[i] = theta[i]; // set the starting point to be theta

  lbfgsfloatval_t objVal = 0.0;
  int ret = -1;
  Optimizer::status status = Optimizer::FAILURE;
  
  for (int t = 0; t < _restarts; t++) {
    lbfgs_parameter_t params = _params; // make a copy, since train() is const
    params.epsilon = tol; // use the tolerance passed to train()
    ret = lbfgs(d, x, &objVal, evaluate, _quiet ? 0 : progress, &inst, &params);
    bool terminate = false;
    if (!_quiet)
      cout << name() << ": ";
    switch (ret) {
      case LBFGSERR_ROUNDING_ERROR:
      case LBFGSERR_MINIMUMSTEP:
      case LBFGSERR_MAXIMUMSTEP:
      case LBFGSERR_INCREASEGRADIENT:
      case LBFGSERR_INVALIDPARAMETERS:
        if (!_quiet)
          cout << "Caught non-convex related error (code " << ret <<
              "). Performing restart.\n";
        status = Optimizer::FAILURE;
        break;
      case LBFGS_CONVERGENCE:
        if (!_quiet)
          cout << "Convergence detected. Performing restart.\n";
        status = Optimizer::CONVERGED;
        break;
      case LBFGSERR_MAXIMUMLINESEARCH:
        if (!_quiet)
          cout << "Reached max number of line search iterations. Terminating\n";
        terminate = true;
        status = Optimizer::FAILURE;
        break;
      case LBFGSERR_MAXIMUMITERATION:
        if (!_quiet)
          cout << "Reached maximum number of iterations. Terminating.\n";
        terminate = true;
        status = Optimizer::MAX_ITERS_CONVEX;
        break;
      case LBFGS_ALREADY_MINIMIZED:
        if (!_quiet)
          cout << "Function appears to be already minimized. Terminating.\n";
        terminate = true;
        status = Optimizer::CONVERGED;
        break;
      default:
        if (!_quiet)
          cout << "lbfgs() returned code " << ret << ". Terminating.\n";
        terminate = true;
        status = Optimizer::FAILURE;
        break;
    }
    if (terminate)
      break;
  }

  // If we've exceeded the maximum number of restarts, we'll say converged.
  if (ret == LBFGSERR_INVALIDPARAMETERS)
    status = Optimizer::CONVERGED;
  
  if (!_quiet) {
    cout << name() << ": Optimization terminated with objective value " <<
        objVal << endl;
    cout << timer.format();
  }
  fval = objVal;
  theta.setWeights(x, d); // copy the final point into theta
  
  // If we evaluated on a validation set, return the best parameters we obtained
  // according to the chosen performance metric. Otherwise, return the current
  // parameters theta.
  if (_validationSetHandler) {
    const Parameters& thetaBest = _validationSetHandler->getBestParams();
    assert(thetaBest.getDimTotal() == d);
    theta.setParams(thetaBest);
    // Return the best validation set performance instead of the objective
    // value, since the former is more useful for model selection.
    fval = _validationSetHandler->getBestScore();
    if (!_quiet) {
      cout << name() << ": Highest performance achieved on validation set was "
          << _validationSetHandler->getBestScore() << " "
          << _validationSetHandler->getPerfMeasure() << endl;
    }
  }
  
  lbfgs_free(x);
  return status;
}
