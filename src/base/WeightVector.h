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

#include <boost/shared_array.hpp>
using boost::shared_array;
#include <string>
using namespace std;
#include "FeatureVector.h"

class RealWeight;

class WeightVector {
  public:
    WeightVector() : _weights(0), _dim(0), _l2(0) {}
    
    WeightVector(int dim);
    
    WeightVector(shared_array<double> weights, int dim);
    
    void reAlloc(int dim);
    
    RealWeight innerProd(const FeatureVector<RealWeight>& fv) const;
    
    // The inner product with a 0 FeatureVector is defined to be zero.
    RealWeight innerProd(const FeatureVector<RealWeight>* fv) const;
    
    void add(const FeatureVector<RealWeight>& fv, const double scale = 1);
    
    void add(const int index, const double value);
    
    double squaredL2Norm() const { return _l2; }
    
    int getDim() const { return _dim; }
    
    const double* getWeights() const { return _weights.get(); }
    
    double getWeight(int index) const { return _weights[index]; }
    
    void setWeights(const double* w, int len);
    
    void zero();
    
    bool read(const string& fname, int dim);
    
    bool write(const string& fname) const;

  private:
  
    shared_array<double> _weights; // the weights
    
    int _dim; // the dimensionality of the vector
    
    double _l2; // the squared L2 norm of weights
    
};
#endif
