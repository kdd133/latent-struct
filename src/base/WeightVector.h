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
    WeightVector() : _dim(0) {}
    
    WeightVector(int dim);
    
    WeightVector(boost::shared_array<double> weights, int dim);
    
    void reAlloc(int dim);
    
    double innerProd(const SparseLogVec& fv) const;
    
    double innerProd(const SparseRealVec& fv) const;
    
    double innerProd(const LogVec& fv) const;
    
    double innerProd(const RealVec& fv) const;
    
    void add(const int index, const double value);
    
    double squaredL2Norm() const;
    
    int getDim() const { return _dim; }
    
    void setWeights(const WeightVector& wv);
    
    void setWeights(const double* w, int len);
    
    void zero();
    
    bool read(const std::string& fname, int dim);
    
    bool write(const std::string& fname) const;
    
    friend std::ostream& operator<<(std::ostream& out, const WeightVector& w);
    
    double operator[](int index) const;
    
    void scale(const double s);
    
    void rescale();
    
    double getScale() const;

  private:
  
    boost::shared_array<double> _weights; // the weights
    
    int _dim; // the dimensionality of the vector
    
    double _scale; // each weight is implicitly multipled by this factor
    
};
#endif
