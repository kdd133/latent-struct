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
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <iostream>
#include <lbfgs.h>

using namespace boost;
using namespace std;

LbfgsOptimizer::LbfgsOptimizer(shared_ptr<TrainingObjective> objective,
                               shared_ptr<Regularizer> regularizer) :
    Optimizer(objective, regularizer), _restarts(3), _minibatchSize(0),
    _maxMinibatches(100), _seed(0), _quiet(false) {
  lbfgs_parameter_init(&_params);
}

int LbfgsOptimizer::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("minibatch-cache", opt::bool_switch(&_minibatchCaching),
        "enable caching of graphs, but only within each minibatch")
    ("m", opt::value<int>(&_params.m)->default_value(20),
        "number of vectors used to compute the approximate the inverse Hessian")
    ("max-iters", opt::value<int>(&_params.max_iterations)->default_value(250),
        "maximum number of iterations (per minibatch, if in minibatch mode)")
    ("max-linesearch",
        opt::value<int>(&_params.max_linesearch)->default_value(5),
        "maximum number of line search iterations")
    ("minibatch-iterations",
        opt::value<int>(&_maxMinibatches)->default_value(10),
        "terminate after processing this many minibatches")
    ("minibatch-size", opt::value<int>(&_minibatchSize)->default_value(0),
        "optimize on a minibatch of this size for several iterations, then \
resample a new minibatch from the training set")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
    ("restarts", opt::value<int>(&_restarts)->default_value(3),
        "number of times to restart Lbfgs when it thinks it has converged")
    ("seed", opt::value<int>(&_seed)->default_value(0),
        "seed for random number generator")
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
  const list<int>* minibatches = inst->mb.get();
  const int batchNum = inst->batchNum;
  double fval = 0;
  
  // Set our model to the current point x.
  theta.setWeights(x, d);
  
  // Compute the gradient at the given theta.
  // Note: The above setting of theta updated the model used here by obj.
  SparseRealVec gradFv(d);
  if (minibatches)
    obj->valueAndGradient(theta, fval, gradFv, &minibatches[batchNum]);
  else
    obj->valueAndGradient(theta, fval, gradFv);
  
  // If the regularizer does not support writing the gradient directly to g,
  // then we copy it manually.
  if (!reg->addRegularization(theta, fval, gradFv, g, d)) {
    reg->addRegularization(theta, fval, gradFv);  
    for (int i = 0; i < d; i++)
      g[i] = gradFv[i];
  }
  
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
  
  if (_minibatchCaching && _objective->getModel(0).getCacheEnabled()) {
    cout << "Error: --minibatch-cache requires that the model cache be " <<
        "disabled (i.e., --cache should not be present)\n";
    return Optimizer::FAILURE;
  }
  
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

  if (_minibatchSize > 0) {
    //// MINIBATCH TRAINING ////
    if (_validationSetHandler) {
      inst.vsh.reset(); // disable validation for the internal solver
      _validationSetHandler->clearBest();
    }
    int numMinibatches = 0;
    inst.batchNum = 0;
    
    for (int b = 1; b <= _maxMinibatches; b++) {
      if (inst.batchNum == 0) {
        // We're at the beginning of the dataset, so sample a new sequence of
        // minibatches.
        numMinibatches = sampleMinibatches(inst.mb, _seed + b);
      }
      assert(numMinibatches > 0);
      assert(inst.batchNum < numMinibatches);
      assert(inst.mb);
      
      if (!_quiet) {
        cout << name() << ": Minibatch iteration " << b << " of " <<
            _maxMinibatches << " (batch index " << inst.batchNum << ", " <<
            inst.mb[inst.batchNum].size() << " examples)\n";
      }
      
      lbfgs_parameter_t params = _params; // make a copy, since train() is const
      params.epsilon = tol; // use the tolerance passed to train()
      
      if (_minibatchCaching) {
        for (size_t mi = 0; mi < _objective->getNumModels(); mi++)
          _objective->getModel(mi).setCacheEnabled(true);
      }
      
      // We call getLbfgsStatus to print some info to stdout, but we ignore
      // the status that's returned, since we're going to optimize on a
      // different minibatch next time in any case.
      ret = lbfgs(d, x, &objVal, evaluate, _quiet ? 0 : progress, &inst, &params);
      getLbfgsStatus(ret, status);
      
      if (_minibatchCaching) {
        // We need to clear the model cache; otherwise, as we process more
        // minibatches, the cache will eventually store all the training
        // examples and we risk running out of memory.
        for (size_t mi = 0; mi < _objective->getNumModels(); mi++) {
          _objective->getModel(mi).emptyCache();
          _objective->getModel(mi).setCacheEnabled(false);
        }
      }
      
      // Evaluate the parameters on the entire validation set. 
      if (_validationSetHandler) {
        const bool caching = _objective->getModel(0).getCacheEnabled();
        if (caching) { // we never want to cache the validation examples
          for (size_t mi = 0; mi < _objective->getNumModels(); mi++)
            _objective->getModel(mi).setCacheEnabled(false);
        }
        _validationSetHandler->evaluate(theta, b);
        if (caching) { // restore caching if it was originally enabled
          for (size_t mi = 0; mi < _objective->getNumModels(); mi++)
            _objective->getModel(mi).setCacheEnabled(true);
        }
      }

      inst.batchNum = (inst.batchNum + 1) % numMinibatches; // next batch
    }
  }
  else {
    //// BATCH TRAINING ////
    for (int t = 0; t < _restarts; t++) {
      lbfgs_parameter_t params = _params; // make a copy, since train() is const
      params.epsilon = tol; // use the tolerance passed to train()
      ret = lbfgs(d, x, &objVal, evaluate, _quiet ? 0 : progress, &inst, &params);
      
      bool terminate = getLbfgsStatus(ret, status);
      if (terminate)
        break;
    }
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
    if (!_validationSetHandler->wasEvaluated()) {
      // It's possible that we never evaluated on the validation set, for
      // example, if L-BFGS's convergence criterion was satisfied by the initial
      // parameters that were given.
      _validationSetHandler->evaluate(theta, 0);
    }
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

bool LbfgsOptimizer::getLbfgsStatus(int ret, Optimizer::status& status)
    const {
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
      return false;
    case LBFGS_CONVERGENCE:
      if (!_quiet)
        cout << "Convergence detected. Performing restart.\n";
      status = Optimizer::CONVERGED;
      return false;
    case LBFGSERR_MAXIMUMLINESEARCH:
      if (!_quiet)
        cout << "Reached max number of line search iterations. Terminating\n";
      status = Optimizer::FAILURE;
      return true;
    case LBFGSERR_MAXIMUMITERATION:
      if (!_quiet)
        cout << "Reached maximum number of iterations. Terminating.\n";
      status = Optimizer::MAX_ITERS_CONVEX;
      return true;
    case LBFGS_ALREADY_MINIMIZED:
      if (!_quiet)
        cout << "Function appears to be already minimized. Terminating.\n";
      status = Optimizer::CONVERGED;
      return true;
    default:
      if (!_quiet)
        cout << "lbfgs() returned code " << ret << ". Terminating.\n";
      status = Optimizer::FAILURE;
      return true;
  }
}

int LbfgsOptimizer::sampleMinibatches(shared_array<list<int> >& minibatches,
    int seed) const {
  assert(_minibatchSize > 0);  
  
  // Get a random ordering for the examples.
  const Dataset& allData = _objective->getDataset(); 
  size_t m = allData.numExamples();
  shared_array<int> ordering = Utility::randPerm(m, seed);
  
  // Group the training examples into minibatches based on the random ordering.
  int numMinibatches = ceil(m / (float)_minibatchSize);
  minibatches.reset(new list<int>[numMinibatches]);
  int j = 0;
  for (int mb = 0; mb < numMinibatches; mb++) {
    for (int count = 0; count < _minibatchSize && j < m; count++)
      minibatches[mb].push_back(ordering[j++]);
  }
  return numMinibatches;
}
