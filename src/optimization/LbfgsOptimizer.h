/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LBFGSOPTIMIZER_H
#define _LBFGSOPTIMIZER_H

#include "Optimizer.h"
#include "Parameters.h"
#include <boost/shared_ptr.hpp>
#include <lbfgs.h>
#include <string>

class Dataset;
class Regularizer;
class TrainingObjective;

class LbfgsOptimizer : public Optimizer {

  public:
    LbfgsOptimizer(boost::shared_ptr<TrainingObjective> objective,
                   boost::shared_ptr<Regularizer> regularizer);
    
    virtual ~LbfgsOptimizer() {}

    virtual Optimizer::status train(Parameters& theta, double& funcVal,
      double tolerance) const;

    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "Lbfgs";
      return _name;
    }
    
  private:
  
    lbfgs_parameter_t _params; // structure that stores L-BFGS options
    
    // Effectively, clear the inverse Hessian approximation the first
    // "restarts" times Lbfgs thinks it has converged. We do this because Lbfgs
    // can be fooled by non-convex objectives, thinking it cannot make further
    // progress when it actually can.
    int _restarts; 
    
    bool _quiet; // suppress optimization output

    static lbfgsfloatval_t evaluate(void* instance, const lbfgsfloatval_t* x,
        lbfgsfloatval_t* g, const int d, const lbfgsfloatval_t step);
        
    static int progress(void* instance, const lbfgsfloatval_t* x,
      const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
      const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
      const lbfgsfloatval_t step, int n, int k, int ls);
        
    typedef struct {
      TrainingObjective* obj;
      Regularizer* reg;
      Parameters* theta;
      bool quiet;
    } LbfgsInstance;
};

#endif
