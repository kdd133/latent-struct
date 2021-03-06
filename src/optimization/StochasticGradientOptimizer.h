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
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
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
    
    virtual bool isOnline() const {
      return true;
    }
    
  private:
    
    std::size_t _maxIters; // maximum number of iterations
    
    bool _autoEta; // if true, estimate the optimal eta on a small data sample
    
    double _eta; // learning rate
    
    // report the average cost after every n updates
    std::size_t _reportAvgCost; 
    
    // report objective value every n updates
    std::size_t _reportObjVal;
    
    bool _quiet; // suppress optimization output
    
    int _seed; // seed for random number generator
    
    // update parameters based on minibatches of this many examples
    std::size_t _minibatchSize;
    
    double objectiveValueForLearningRate(const Parameters& theta,
        const std::list<int>& sample, const std::list<int>* minibatches,
        std::size_t numMinibatches, double eta) const;
      
    double objectiveValueForSample(const Parameters& theta,
        const std::list<int>& sample) const;
        
    double estimateBestLearningRate(const Parameters& theta,
        const std::list<int>& sample, double eta0) const;
};

#endif
