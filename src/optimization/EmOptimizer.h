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
#include <boost/shared_ptr.hpp>
using namespace boost;

class WeightVector;
class Dataset;
class TrainingObjective;

class EmOptimizer : public Optimizer {

  public:
    EmOptimizer(TrainingObjective& objective, shared_ptr<const Optimizer> opt);
    
    virtual ~EmOptimizer() {}

    virtual double train(WeightVector& w) const;

    virtual int processOptions(int argc, char** argv);
    
    static const string& name() {
      static const string _name = "EM";
      return _name;
    }
    
  private:
  
    shared_ptr<const Optimizer> _convexOpt; // the convex optimization procedure
  
    double _epsilon; // value used when testing for convergence
    
    int _maxIters; // maximum number of iterations
    
    bool _quiet; // suppress optimization output
};

#endif
