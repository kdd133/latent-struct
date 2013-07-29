/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _BMRMOPTIMIZER_H
#define _BMRMOPTIMIZER_H

#include "Optimizer.h"
#include "Parameters.h"
#include <boost/shared_ptr.hpp>
#include <string>

class Dataset;
class Regularizer;
class TrainingObjective;

class BmrmOptimizer : public Optimizer {

  public:
    BmrmOptimizer(boost::shared_ptr<TrainingObjective> objective,
                  boost::shared_ptr<Regularizer> regularizer);
    
    virtual ~BmrmOptimizer() {}

    virtual Optimizer::status train(Parameters& theta, double& funcVal,
      double tolerance) const;

    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "Bmrm";
      return _name;
    }
    
  private:
    
    std::size_t _maxIters; // maximum number of iterations
    
    bool _quiet; // suppress optimization output
    
    bool _noShrinking; // if true, disable the shrinking heuristic
    
    // the performance measure that determines the "best" set of parameters
    std::string _perfMeasure;
};

#endif
