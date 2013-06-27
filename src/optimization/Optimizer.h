/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include "Parameters.h"
#include "Regularizer.h"
#include <boost/shared_ptr.hpp>

class Dataset;
class TrainingObjective;

class Optimizer {

  public:
    Optimizer(boost::shared_ptr<TrainingObjective> objective,
              boost::shared_ptr<Regularizer> regularizer)
      : _objective(objective), _regularizer(regularizer) {}
    
    virtual ~Optimizer() {}
    
    enum status {
      // The optimizer converged.
      CONVERGED,
      // The max number of iterations was reached while directly optimizing an
      // objective (usually convex) function.
      MAX_ITERS_CONVEX,
      // The alternating optimizer reached its maximum number of iterations.
      MAX_ITERS_ALTERNATING,
      // The optimizer made negative progress from some previous iterate.
      BACKWARD_PROGRESS,
      // A general failure occurred.
      FAILURE
    };

    // Returns a status code; stores the objective value at the optimal point
    // in funcVal.
    virtual Optimizer::status train(Parameters& theta, double& funcVal,
      double tolerance) const = 0;

    virtual int processOptions(int argc, char** argv) = 0;
    
    virtual bool usesValidationSet() const {
      return false;
    }
    
    virtual bool isOnline() const {
      return false; // assume the default is batch learning
    }
    
    virtual void setValidation(const boost::shared_ptr<Dataset>& val) {
      _validationSet = val;
    }
    
  protected:
  
    boost::shared_ptr<TrainingObjective> _objective;
    
    boost::shared_ptr<Regularizer> _regularizer;
    
    boost::shared_ptr<Dataset> _validationSet;
};

#endif
