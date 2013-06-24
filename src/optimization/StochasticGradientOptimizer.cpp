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
#include "WeightVector.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
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
    _reportAvgCost(100), _reportValStats(1000), _quiet(false), _seed(0),
    _valSetFraction(0.1), _threads(1) {
}

int StochasticGradientOptimizer::processOptions(int argc, char** argv) {
  namespace opt = program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("learning-rate", opt::value<double>(&_eta)->default_value(0.01),
        "the learning rate")
    ("max-iters", opt::value<size_t>(&_maxIters)->default_value(250),
        "maximum number of iterations")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
    ("report-avg-cost", opt::value<size_t>(&_reportAvgCost)->
        default_value(100), "report the avg. cost every n updates")
    ("report-validation-stats", opt::value<size_t>(&_reportValStats)->
        default_value(1000), "report validation performance every n updates")
    ("seed", opt::value<int>(&_seed)->default_value(0),
        "seed for random number generator")
    ("threads", opt::value<size_t>(&_threads)->default_value(1),
        "number of threads used to parallelize computing validation set score")
    ("fraction-validation", opt::value<double>(&_valSetFraction)->default_value(
        0.1), "fraction of training examples to use as a validation set")
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
  size_t m = allData.numExamples();
  const double beta = _regularizer->getBeta();
  const size_t d = theta.getDimTotal();
  assert(d > 0);

  // Get a random ordering for the examples.
  shared_array<int> ordering = Utility::randPerm(m, _seed);
  
  // If a distinct evaluation set has not been provided (to the superclass),
  // then split the training data into smaller training and validation sets.
  boost::shared_ptr<Dataset> validationData;
  if (!_validationSet) {
    validationData.reset(new Dataset(_threads));
    const size_t mAll = m;
    m -= m * _valSetFraction;
    // Form the validation set using examples with indices[m,...,mAll-1].
    for (size_t i = m; i < mAll; ++i)
      validationData->addExample(allData.getExamples()[ordering[i]]);
  }
  else
    validationData = _validationSet;
    
  // Create a data structure that will be used to store the predictions made on
  // the validation set. 
  size_t maxId = 0;
  BOOST_FOREACH(const Example& ex, validationData->getExamples()) {
    const size_t id = ex.x()->getId();
    if (id > maxId)
      maxId = id;
  }  
  LabelScoreTable labelScores(maxId + 1, allData.getLabelSet().size());
  
  double cost = 0;
  RealVec grad(d);
  
  // This variable will store a running total of the costs, summed over the last
  // R examples that we've seen/processed.
  double sumCosts = 0;
  
  double fscoreBest = -1;
  Parameters thetaBest;
  
  // FIXME: We're assuming L2 regularization (ignoring _regularizer) below.
  size_t nUpdates = 0;
  for (size_t t = 0; t < _maxIters; ++t)
  {
    timer::cpu_timer timer;
    
    for (size_t i = 0; i < m; ++i) {
      // compute the gradient and the cost function value for this example
      // (note: the cost returned here does not account for regularization)
      _objective->valueAndGradientOne(theta, cost, grad, ordering[i]);
      sumCosts += cost;
      
      // update the parameters based on the computed gradient
      for (size_t j = 0; j < d; ++j) {
        theta.add(j, -_eta * (beta*theta[j] + grad[j]));
      }
      nUpdates++;
      
      // Report the average cost (plus regularization) taken over the last n
      // updates, where the cost of each example is computed prior to the
      // parameter update (gradient step).
      if (!_quiet && nUpdates % _reportAvgCost == 0) {
        const size_t R = _reportAvgCost;
        costLastR = 0.5 * beta * theta.squaredL2Norm() + (sumCosts / R);
        printf("%s: t = %d  nU = %d  cost_last_%d = %.5f\n", name().c_str(),
            (int)t, (int)nUpdates, (int)_reportAvgCost, costLastR);
        sumCosts = 0; // reset the running total
      }
      
      // Evaluate the performance of model on the held-out data.
      if (nUpdates % _reportValStats == 0) {
        double accuracy, precision, recall, fscore;
        _objective->predict(theta, *validationData, labelScores);
        Utility::calcPerformanceMeasures(*validationData, labelScores, false,
            "", "", accuracy, precision, recall, fscore);
          
        if (!_quiet) {
          printf("%s: t = %d  nU = %d  acc = %.3f  prec = %.3f  rec = %.3f  ",
              name().c_str(), (int)t, (int)nUpdates, accuracy, precision,
              recall);
          printf("fscore = %.3f  timer: %s" , fscore, timer.format().c_str());
        }
    
        if (fscore > fscoreBest) {
          fscoreBest = fscore;
          thetaBest.setParams(theta);
        }
      }
    }
  }

  theta.setParams(thetaBest);
  
  // We don't actually test for convergence (simply run for the specified number
  // of epochs); so, we'll just call it converged.
  return Optimizer::CONVERGED;
}
