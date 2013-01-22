/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _EMOPTIMIZER_H
#define _EMOPTIMIZER_H

#include "Optimizer.h"
#include "Parameters.h"
#include <boost/shared_ptr.hpp>
#include <string>

class Dataset;
class Regularizer;
class TrainingObjective;

class EmOptimizer : public Optimizer {

  public:
    EmOptimizer(boost::shared_ptr<TrainingObjective> objective,
                boost::shared_ptr<Regularizer> regularizer,
                boost::shared_ptr<Optimizer> opt);
    
    virtual ~EmOptimizer() {}

    virtual Optimizer::status train(Parameters& theta, double& funcVal,
      double tolerance) const;

    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "EM";
      return _name;
    }
    
  private:
  
    boost::shared_ptr<Optimizer> _convexOpt; // the convex optimizer
  
    int _maxIters; // maximum number of iterations
    
    // If true, we abort if the inner solver exceeds its maximum number of
    // iterations on two consecutive EM iterations.
    bool _abortOnConsecMaxIters;
    
    bool _quiet; // suppress optimization output
};

#endif
