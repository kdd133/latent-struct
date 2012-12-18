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

class Dataset;
class TrainingObjective;

class Optimizer {

  public:
    Optimizer(TrainingObjective& objective, double beta)
      : _objective(objective), _beta(beta) {}
    
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
    
    double getBeta() const;
    
    virtual void setBeta(double beta);
    
  protected:
  
    TrainingObjective& _objective;
    
    double _beta; // regularization constant

};

inline double Optimizer::getBeta() const {
  return _beta;
}

inline void Optimizer::setBeta(double beta) {
  _beta = beta;
}

#endif
