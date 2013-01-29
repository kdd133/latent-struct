/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _WEIGHTVECTOR_H
#define _WEIGHTVECTOR_H

#include "Ublas.h"
#include <boost/shared_array.hpp>
#include <ostream>
#include <string>

class LogWeight;

class WeightVector {
  
  public:
    WeightVector() : _weights(0), _dim(0), _l2(0) {}
    
    WeightVector(int dim);
    
    WeightVector(boost::shared_array<double> weights, int dim);
    
    void reAlloc(int dim);
    
    double innerProd(const SparseLogVec& fv) const;
    
    double innerProd(const SparseRealVec& fv) const;
    
    double innerProd(const LogVec& fv) const;
    
    double innerProd(const RealVec& fv) const;
    
    void add(const int index, const double value);
    
    double squaredL2Norm() const { return _l2; }
    
    int getDim() const { return _dim; }
    
    const double* getWeights() const { return _weights.get(); }
    
    double getWeight(int index) const { return _weights[index]; }
    
    void setWeights(const double* w, int len);
    
    void zero();
    
    bool read(const std::string& fname, int dim);
    
    bool write(const std::string& fname) const;
    
    friend std::ostream& operator<<(std::ostream& out, const WeightVector& w);
    
    const double& operator[](int index) const;

  private:
  
    boost::shared_array<double> _weights; // the weights
    
    int _dim; // the dimensionality of the vector
    
    double _l2; // the squared L2 norm of weights
    
};
#endif
