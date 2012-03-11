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
    Optimizer(TrainingObjective& objective, double beta)
      : _objective(objective), _beta(beta) {}
    
    virtual ~Optimizer() {}

    // Returns the objective value at the optimal point, or infinity if an
    // error was encountered.
    virtual double train(WeightVector& w, double tolerance) const = 0;

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
