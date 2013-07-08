/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _REGULARIZERNONE_H
#define _REGULARIZERNONE_H

#include "Parameters.h"
#include "Regularizer.h"
#include "Ublas.h"

/*
 * This class can be used by an Optimizer that either a) does not use any
 * regularization, or b) performs some sort of custom regularization. In the
 * latter case, the beta value stored here may be used by the Optimizer.
 */
class RegularizerNone : public Regularizer {

  public:

    RegularizerNone(double beta = 1e-4) : Regularizer(beta) { }

    virtual ~RegularizerNone() {}

    virtual void addRegularization(const Parameters& theta, double& fval,
        SparseRealVec& grad) const {
      /** do nothing **/
    }
    
    static const std::string& name() {
      static const std::string _name = "None";
      return _name;
    }
};

#endif
