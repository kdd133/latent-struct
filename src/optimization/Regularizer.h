/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _REGULARIZER_H
#define _REGULARIZER_H

#include "Parameters.h"
#include "Ublas.h"

class Regularizer {

  public:

    Regularizer(double beta = 1e-4) : _beta(beta) { }

    virtual ~Regularizer() {}

    virtual void addRegularization(const Parameters& theta, double& fval,
        RealVec& grad) const = 0;
        
    double getBeta() const {
      return _beta;
    }
    
    void setBeta(double beta) {
      _beta = beta;
    }
  
  protected:
  
    double _beta;
};

#endif
