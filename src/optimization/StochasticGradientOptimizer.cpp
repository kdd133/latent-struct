/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
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
#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_deque.hpp>
#include <boost/shared_array.hpp>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <string>
#include <uQuadProg++.hh>

using namespace boost;
using namespace std;

StochasticGradientOptimizer::StochasticGradientOptimizer(shared_ptr<TrainingObjective> objective,
                             shared_ptr<Regularizer> regularizer) :
    Optimizer(objective, regularizer), _maxIters(250), _quiet(false) {
}

int StochasticGradientOptimizer::processOptions(int argc, char** argv) {
  namespace opt = program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("max-iters", opt::value<size_t>(&_maxIters)->default_value(250),
        "maximum number of iterations")
//    ("no-shrinking", opt::bool_switch(&_noShrinking),
//        "disable the shrinking heuristic")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
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
  
  const size_t T = 1000; // maximum number of epochs (passes over the data)
  double eta = 0.0025;  // learning rate
  
  // The idea of the "patience" stopping heuristic is adapted from
  // http://deeplearning.net/tutorial/mlp.html
  // process at least this many examples
  size_t patience = 1000;
  // when a new minimizer is found, wait this much longer //TODO: clarify this
  const size_t patienceIncrease = 2;
  // increase the patience if a relative improvement of this amount is observed
  const double improvementThreshold = 0.9999;
  
  // Each time we've seen a multiple of R examples, report the average cost
  // (plus regularization) taken over the last R updates, where the cost of
  // each example is computed prior to the parameter update (gradient step).
  const size_t R = 100;
  
  const size_t m = _objective->getDataset().numExamples();
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
  bool guessConverged = false;
  double lowestCost = std::numeric_limits<double>::infinity();

  // Get a random ordering for the training examples.
  shared_array<int> ordering = Utility::randPerm(m);
  
  for (size_t t = 0; t < T && !guessConverged; ++t)
  {
//    timer::cpu_timer timer;
    
    for (size_t i = 0; i < m; ++i) {
      // compute the gradient and the cost function value for this example
      // (note: the cost returned here does not account for regularization)
      _objective->valueAndGradientOne(theta, cost, grad, ordering[i]);
      sumCosts += cost;
      
      // update the parameters based on the computed gradient
      for (size_t j = 0; j < d; ++j) {
        theta.add(j, -eta * (beta*theta[j] + grad[j]));
      }
      
      if (++numExamplesSeen % R == 0) {
        costLastR = 0.5 * beta * theta.squaredL2Norm() + (sumCosts / R);
        cout << name() << " t = " << t << "  fval_last_" << R << " = "
            << costLastR << endl;
        sumCosts = 0; // reset the running total
        
        // FIXME: It doesn't really make sense to check this on the training
        // data, since we are only considering the last R examples. Thus, if
        // a particular sequence of R examples are especially "easy", costLastR
        // will be unusually low, even though the overall objective value may
        // not have improved proportionally.
        if (costLastR < lowestCost) {
          if (costLastR < lowestCost * improvementThreshold) 
            patience = std::max(patience, numExamplesSeen * patienceIncrease);
            
          lowestCost = costLastR; 
        }
      }
      
      if (numExamplesSeen >= patience) {
        guessConverged = true;
        break;
      }
    }
  }

  return Optimizer::CONVERGED;
}
