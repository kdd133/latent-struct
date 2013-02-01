/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _REGULARIZERL2_H
#define _REGULARIZERL2_H

#include "Parameters.h"
#include "Regularizer.h"
#include "Ublas.h"

class RegularizerL2 : public Regularizer {

  public:

    RegularizerL2(double beta = 1e-4);

    virtual ~RegularizerL2() {}

    virtual void addRegularization(const Parameters& theta, double& fval,
        RealVec& grad) const;
        
    static const std::string& name() {
      static const std::string _name = "L2";
      return _name;
    }
    
    virtual int processOptions(int argc, char** argv);
    
  private:
  
    // Note: The _beta value (from the parent class) will be ignored if either
    // _betaW > 0 or _betaU > 0. If _betaW == 0 and _betaU > 0, the w parameters
    // will not be regularized, and vice versa.
    double _betaW;
    double _betaU;
};

#endif
