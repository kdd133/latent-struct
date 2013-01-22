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

    RegularizerL2(double beta = 1e-4) : Regularizer(beta) { }

    virtual ~RegularizerL2() {}

    virtual void addRegularization(const Parameters& theta, double& fval,
        RealVec& grad) const;
};

#endif
