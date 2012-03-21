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
#include <string>

class Dataset;
class TrainingObjective;
class WeightVector;

class BmrmOptimizer : public Optimizer {

  public:
    BmrmOptimizer(TrainingObjective& objective);
    
    virtual ~BmrmOptimizer() {}

    virtual Optimizer::status train(WeightVector& w, double& funcVal,
      double tolerance) const;

    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "Bmrm";
      return _name;
    }
    
  private:
    
    size_t _maxIters; // maximum number of iterations
    
    bool _quiet; // suppress optimization output
    
    bool _noShrinking; // if true, disable the shrinking heuristic
};

#endif
