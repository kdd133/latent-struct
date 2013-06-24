/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Kenneth Dwyer
 */

#ifndef _STOCHASTICGRADIENTOPTIMIZER_H
#define _STOCHASTICGRADIENTOPTIMIZER_H

#include "Optimizer.h"
#include "Parameters.h"
#include <boost/shared_ptr.hpp>
#include <string>

class Dataset;
class Regularizer;
class TrainingObjective;

class StochasticGradientOptimizer : public Optimizer {

  public:
    StochasticGradientOptimizer(boost::shared_ptr<TrainingObjective> objective,
                  boost::shared_ptr<Regularizer> regularizer);
    
    virtual ~StochasticGradientOptimizer() {}

    virtual Optimizer::status train(Parameters& theta, double& funcVal,
      double tolerance) const;

    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "StochasticGradient";
      return _name;
    }
    
    virtual bool usesValidationSet() const {
      return true;
    }
    
  private:
    
    std::size_t _maxIters; // maximum number of iterations
    
    double _eta; // learning rate
    
    // print a progress report after this many updates
    std::size_t _progressReportUpdates; 
    
    bool _quiet; // suppress optimization output
    
    int _seed; // seed for random number generator
    
    // fraction of examples to use as validation set (if a distinct validation
    // set is not provided)
    double _valSetFraction;
    
    // # of threads used to parallelize computing validation set score 
    std::size_t _threads;
};

#endif
