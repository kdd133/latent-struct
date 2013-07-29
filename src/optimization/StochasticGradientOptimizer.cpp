/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Kenneth Dwyer
 */


#include "Dataset.h"
#include "Model.h"
#include "Optimizer.h"
#include "Regularizer.h"
#include "StochasticGradientOptimizer.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include "Utility.h"
#include "ValidationSetHandler.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <list>
#include <stdio.h>
#include <string>

using namespace boost;
using namespace std;

StochasticGradientOptimizer::StochasticGradientOptimizer(
    shared_ptr<TrainingObjective> objective, shared_ptr<Regularizer> regularizer) :
    Optimizer(objective, regularizer), _maxIters(250), _autoEta(false),
    _eta(0.01), _reportAvgCost(100), _reportValStats(1000), _reportObjVal(true),
    _quiet(false), _seed(0), _minibatchSize(1) {
}

int StochasticGradientOptimizer::processOptions(int argc, char** argv) {
  namespace opt = program_options;
  bool noObjVal = false;
  opt::options_description options(name() + " options");
  options.add_options()
    ("estimate-learning-rate", opt::bool_switch(&_autoEta),
        "estimate the optimal learning rate on a sample of 1000 examples")
    ("learning-rate", opt::value<double>(&_eta)->default_value(0.01),
        "the learning rate")
    ("max-iters", opt::value<size_t>(&_maxIters)->default_value(250),
        "maximum number of iterations")
    ("minibatch-size", opt::value<size_t>(&_minibatchSize)->default_value(1),
        "update parameters based on minibatches of this many examples")
    ("no-report-objective-value", opt::bool_switch(&noObjVal),
        "do not compute/report the objective value along with validation stats")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
    ("report-avg-cost", opt::value<size_t>(&_reportAvgCost)->
        default_value(100), "report the avg. cost every n updates")
    ("report-validation-stats", opt::value<size_t>(&_reportValStats)->
        default_value(1000), "report validation performance every n updates")
    ("seed", opt::value<int>(&_seed)->default_value(0),
        "seed for random number generator")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (noObjVal)
    _reportObjVal = false;
  
  if (vm.count("help"))
    cout << options << endl;

  return 0;
}

Optimizer::status StochasticGradientOptimizer::train(Parameters& theta,
    double& bestPerf, double tol) const {
  
  const Dataset& allData = _objective->getDataset(); 
  size_t m = allData.numExamples();
  const double beta = _regularizer->getBeta();
  const size_t d = theta.getDimTotal();
  assert(d > 0);

  // Get a random ordering for the examples.
  shared_array<int> ordering = Utility::randPerm(m, _seed);
  
  // This variable is used to simplify some of the if statements below.
  const bool doValid = _validationSetHandler && _reportValStats > 0;
  
  // Group the training examples into minibatches based on the random ordering.
  int numMinibatches = ceil(m / (float)_minibatchSize);
  scoped_array<list<int> > minibatches(new list<int>[numMinibatches]);
  int j = 0;
  for (int mb = 0; mb < numMinibatches; mb++) {
    for (int count = 0; count < _minibatchSize && j < m; count++)
      minibatches[mb].push_back(ordering[j++]);
  }
  
  double eta0 = _eta;
  if (_autoEta) {
    int sampleSize = 1000;
    if (sampleSize > m)
      sampleSize = m;
    list<int> sample;
    for (int j = 0; j < sampleSize; j++)
      sample.push_back(ordering[j]);
    eta0 = estimateBestLearningRate(theta, sample, 0.1);
  }
  if (!_quiet)
    cout << "Using learning rate eta0 = " << eta0 << endl;
  double eta_t = eta0;
  
  ordering.reset(); // we can safely discard this
  
  double cost = 0;
  SparseRealVec grad(d);
  
  // This variable will store a running total of the costs, summed over the last
  // R examples that we've seen/processed.
  double sumCosts = 0;
  
  // FIXME: We're assuming L2 regularization (ignoring _regularizer) below.
  size_t t = 0;
  for (size_t ep = 0; ep < _maxIters; ++ep)
  {
    timer::cpu_timer timer;
    
    for (size_t i = 0; i < numMinibatches; ++i) {
      // compute the gradient and the cost function value for this example
      // (note: the cost returned here does not account for regularization)
      if (_minibatchSize > 1) {
        _objective->valueAndGradient(theta, cost, grad, &minibatches[i]);
      }
      else {
        // avoid multi-threading overhead in this case
        _objective->valueAndGradientOne(theta, cost, grad,
            minibatches[i].front());
      }        
      sumCosts += cost;
      
      // update the parameters based on the computed gradient; see Sec 5.1 of
      // http://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf
      // note: the scaling of theta must be done before the add operations
      theta.scale(1 - beta * eta_t);
      SparseRealVec::const_iterator it;
      for (it = grad.begin(); it != grad.end(); ++it)
        theta.add(it.index(), -eta_t * (*it));
      t++;
      
      eta_t = eta0 / (1 + eta0*beta*t); // update the learning rate
      
      // Report the average cost (plus regularization) taken over the last n
      // updates, where the cost of each example is computed prior to the
      // parameter update (gradient step).
      if (!_quiet && _reportAvgCost > 0 && t % _reportAvgCost == 0) {
        const size_t R = _reportAvgCost;
        double costLastR = 0.5 * beta * theta.squaredL2Norm() + (sumCosts / R);
        printf("%s: ep = %d  t = %d  cost_last_%d = %.5f  time_last_%d =%s",
            name().c_str(), (int)ep, (int)t, (int)_reportAvgCost, costLastR,
            (int)_reportAvgCost, timer.format().c_str());
        sumCosts = 0; // reset the running total
        timer.start();
      }
      
      // Compute and report the actual objective value.
      if (_reportObjVal && !_quiet) {
        timer.stop();
        timer::cpu_timer clock;
        cout << name() << ": Computing objective value...  ";
        cout.flush();
        double fval;
        SparseRealVec gradTemp(d);
        _objective->valueAndGradient(theta, fval, gradTemp);
        printf("obj = %.5f ", fval);
        cout << clock.format();
        timer.resume();
      }
      
      // Evaluate the performance of model on the held-out data. This is done
      // every _reportValStats updates, and after the first update.
      if (doValid && (t == 1 || t % _reportValStats == 0)) {
        timer.stop();
        assert(_validationSetHandler);
        _validationSetHandler->evaluate(theta, t);
        timer.resume();
      }
    }
  }

  // If we evaluated on a validation set, return the best set of parameters we
  // found. Otherwise, return the current parameters theta.
  bool havePerformanceStatistic = false;
  if (doValid) {
    assert(_validationSetHandler);
    const Parameters& thetaBest = _validationSetHandler->getBestParams();
    // If predictions were never actually made on the validation set (due to
    // not enough iterations, for example), then the dimensionality of
    // thetaBest will be zero, as per its initializaiton.
    if (thetaBest.getDimWU() > 0) {
      theta.setParams(thetaBest);
      // Return the best validation set performance instead of the objective
      // value, since the former is more useful for model selection.
      bestPerf = _validationSetHandler->getBestScore();
      if (!_quiet) {
        cout << name() << ": Highest performance achieved on validation set was "
            << _validationSetHandler->getBestScore() << " "
            << _validationSetHandler->getPerfMeasure() << endl;
      }
      havePerformanceStatistic = true;
    }
  }
  
  // If we didn't compute any performance statistic, then return the objective
  // value instead.
  if (!havePerformanceStatistic) {
    assert(!doValid); // we shouldn't arrive here if we used a validation set
    timer::cpu_timer clock;
    if (!_quiet) {
      cout << name() << ": Computing objective value...  ";
      cout.flush();
    }
    SparseRealVec gradTemp(d);
    _objective->valueAndGradient(theta, bestPerf, gradTemp);
    if (!_quiet) {
      printf("obj = %.5f ", bestPerf);
      cout << clock.format();
    }
  }
  
  // We don't actually test for convergence (simply run for the specified number
  // of epochs); so, we'll just call it converged.
  return Optimizer::CONVERGED;
}

double StochasticGradientOptimizer::objectiveValueForSample(
    const Parameters& theta, const list<int>& sample) const {
  // Compute the objective value with theta for the given sample.
  SparseRealVec grad(theta.getDimTotal());
  double avgCost;
  _objective->valueAndGradient(theta, avgCost, grad, &sample);
  return 0.5 * _regularizer->getBeta() * theta.squaredL2Norm() + avgCost;
}

// Based on code in crfsgd.cpp by Leon Bottou:
// http://leon.bottou.org/projects/sgd
double StochasticGradientOptimizer::objectiveValueForLearningRate(
    const Parameters& theta_, const list<int>& sample,
    const list<int>* minibatches, size_t numMinibatches, double eta) const {
  Parameters theta(theta_.getDimW(), theta_.getDimU());
  theta.setParams(theta_);
  
  const double beta = _regularizer->getBeta();
  const size_t d = theta.getDimTotal();
  SparseRealVec grad(d);
  double cost;

  // Perform one epoch of learning using the given sample.
  for (size_t i = 0; i < numMinibatches; ++i) {
    if (_minibatchSize > 1) {
      _objective->valueAndGradient(theta, cost, grad, &minibatches[i]);
    }
    else {
      // avoid multi-threading overhead in this case
      _objective->valueAndGradientOne(theta, cost, grad,
          minibatches[i].front());
    }        
  
    theta.scale(1 - beta * eta);
    SparseRealVec::const_iterator it;
    for (it = grad.begin(); it != grad.end(); ++it)
      theta.add(it.index(), -eta * (*it));
  }
  
  return objectiveValueForSample(theta, sample);
}

// Based on code in crfsgd.cpp by Leon Bottou:
// http://leon.bottou.org/projects/sgd
double StochasticGradientOptimizer::estimateBestLearningRate(
    const Parameters& theta, const list<int>& sample, double eta0) const {
  timer::cpu_timer timer;
  size_t numSamples = sample.size();
  if (!_quiet) {
    cout << "Estimating best learning rate based on " << numSamples <<
      " samples" << endl;
  }
  size_t numExamples = _objective->getDataset().numExamples();
  if (numSamples > numExamples)
    numSamples = numExamples;
    
  // Group the examples into minibatches.
  int numMinibatches = ceil(numSamples / (float)_minibatchSize);
  scoped_array<list<int> > minibatches(new list<int>[numMinibatches]);
  int count = 0;
  BOOST_FOREACH(int i, sample)
    minibatches[count++ % numMinibatches].push_back(i);

  double obj0 = objectiveValueForSample(theta, sample);
  if (!_quiet)
    cout << "  Initial objective value: " << obj0 << endl;

  // empirically find eta that works best
  double bestEta = 1;
  double bestObj = obj0;
  double eta = eta0;
  int toTest = 10;
  double factor = 2;
  bool phase2 = false;
  while (toTest > 0 || !phase2) {
    double obj = objectiveValueForLearningRate(theta, sample, minibatches.get(),
        numMinibatches, eta);
    bool okay = (obj < obj0);
    if (!_quiet) {
      cout << "  Trying eta=" << eta << "  obj=" << obj;
      if (okay)
        cout << " (possible)" << endl;
      else
        cout << " (too large)" << endl;
    }
    if (okay) {
      toTest -= 1;
      if (obj < bestObj) {
        bestObj = obj;
        bestEta = eta;
      }
    }
    if (!phase2) {
      if (okay)
        eta = eta * factor;
      else {
        phase2 = true;
        eta = eta0;
      }
    }
    if (phase2)
      eta = eta / factor;
  }
  if (!_quiet)
    cout << "  Elapsed time: " << timer.format();
  // To be safe, we choose a learning rate that's slightly lower than the best.
  return bestEta / factor;
}
