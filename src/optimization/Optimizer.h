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

class Dataset;
class TrainingObjective;
class WeightVector;

class Optimizer {

  public:
    Optimizer(TrainingObjective& objective, double beta = 1.0)
      : _objective(objective), _beta(beta) {}
    
    virtual ~Optimizer() {}

    // Returns the objective value at the optimal point, or infinity if an
    // error was encountered.
    virtual double train(WeightVector& w) const = 0;

    virtual int processOptions(int argc, char** argv) = 0;
    
    double getBeta() const { return _beta; }

  protected:
  
    TrainingObjective& _objective;
    
    double _beta; // regularization constant

};

#endif
