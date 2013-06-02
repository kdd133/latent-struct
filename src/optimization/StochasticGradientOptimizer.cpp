/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Kenneth Dwyer
 */

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "StochasticGradientOptimizer.h"
#include "Model.h"
#include "Optimizer.h"
#include "Regularizer.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include "Utility.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_array.hpp>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <string>

using namespace boost;
using namespace std;

StochasticGradientOptimizer::StochasticGradientOptimizer(shared_ptr<TrainingObjective> objective,
                             shared_ptr<Regularizer> regularizer) :
    Optimizer(objective, regularizer), _maxIters(250), _eta(0.01),
    _progressReportUpdates(1000), _quiet(false), _seed(0),
    _valSetFraction(0.1) {
}

int StochasticGradientOptimizer::processOptions(int argc, char** argv) {
  namespace opt = program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("learning-rate", opt::value<double>(&_eta)->default_value(0.01),
        "the learning rate")
    ("max-iters", opt::value<size_t>(&_maxIters)->default_value(250),
        "maximum number of iterations")
    ("progress-updates", opt::value<size_t>(&_progressReportUpdates)->
        default_value(1000), "print a progress report after this many updates")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
    ("seed", opt::value<int>(&_seed)->default_value(0),
        "seed for random number generator")
    ("validation", opt::value<double>(&_valSetFraction)->default_value(0.1),
        "fraction of training examples to use as a validation set")
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

Optimizer::status StochasticGradientOptimizer::train(Parameters& theta,
    double& costLastR, double tol) const {
  
  const Dataset& allData = _objective->getDataset(); 
  const size_t m = allData.numExamples();
  const double beta = _regularizer->getBeta();
  const size_t d = theta.getDimTotal();
  assert(d > 0);
  
  // FIXME: We're assuming L2 regularization (ignoring _regularizer) below.
  
  double cost = 0;
  RealVec grad(d);
  
  // This variable will store a running total of the costs, summed over the last
  // R examples that we've seen/processed.
  double sumCosts = 0;
  
  size_t numExamplesSeen = 0;

  // Get a random ordering for the examples.
  shared_array<int> ordering = Utility::randPerm(m, _seed);
  
  // Split the data into training and validation sets.
  const size_t mVal = m * _valSetFraction;
  const size_t mTrain = m - mVal;
  
  // Form the validation set using examples with indices[mTrain,...,m-1].
  Dataset validationData;
  for (size_t i = mTrain; i < m; ++i)
    validationData.addExample(allData.getExamples()[ordering[i]]);
    
  // Create a data structure that will be used to store the predictions made on
  // the validation set. 
  size_t maxId = 0;
  BOOST_FOREACH(const Example& ex, validationData.getExamples()) {
    const size_t id = ex.x()->getId();
    if (id > maxId)
      maxId = id;
  }  
  LabelScoreTable labelScores(maxId + 1, allData.getLabelSet().size());
  
  double accuracyPrevEpoch = -1;
  Parameters thetaPrevEpoch;
  
  for (size_t t = 0; t < _maxIters; ++t)
  {
    timer::cpu_timer timer;
    
    for (size_t i = 0; i < mTrain; ++i) {
      // compute the gradient and the cost function value for this example
      // (note: the cost returned here does not account for regularization)
      _objective->valueAndGradientOne(theta, cost, grad, ordering[i]);
      sumCosts += cost;
      
      // update the parameters based on the computed gradient
      for (size_t j = 0; j < d; ++j) {
        theta.add(j, -_eta * (beta*theta[j] + grad[j]));
      }
      
      // Each time we've seen a multiple of R examples, report the average cost
      // (plus regularization) taken over the last R updates, where the cost of
      // each example is computed prior to the parameter update (gradient step).
      if (!_quiet && ++numExamplesSeen % _progressReportUpdates == 0) {
        const size_t R = _progressReportUpdates;
        costLastR = 0.5 * beta * theta.squaredL2Norm() + (sumCosts / R);
        cout << name() << ": t = " << t << "  fval_last_" << R << " = "
            << costLastR << endl;
        sumCosts = 0; // reset the running total
      }
    }

    // Evaluate the performance of model on the held-out data.
    double accuracy, precision, recall, fscore;
    _objective->predict(theta, validationData, labelScores);
    Utility::calcPerformanceMeasures(validationData, labelScores, false, "", "",
      accuracy, precision, recall, fscore);
      
    if (!_quiet) {
      printf("%s: t = %d  acc = %.3f  prec = %.3f  rec = %.3f  fscore = %.3f",
          name().c_str(), (int)t, accuracy, precision, recall, fscore);
      cout << "  timer:" << timer.format();
    }
    
    // If the accuracy after the current epoch is lower than that of the
    // previous epoch, restore the previous parameters and say converged.
    if (accuracy < accuracyPrevEpoch) {
      theta.setParams(thetaPrevEpoch);
      // TODO: Compute the objective value? (costLastR is fairly meaningless)
      return Optimizer::CONVERGED;
    }
    else if (accuracy - accuracyPrevEpoch < tol) {
      // Here, we treat tol as the minimum amount by which accuracy must
      // increase in order to justify continued optimization.
      // TODO: Compute the objective value? (costLastR is fairly meaningless)
      return Optimizer::CONVERGED;
    }

    // Record the current accuracy and parameters.
    accuracyPrevEpoch = accuracy;
    thetaPrevEpoch.setParams(theta);
  }

  return Optimizer::CONVERGED;
}
