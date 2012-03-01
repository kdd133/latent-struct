/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _UTILITY_H
#define _UTILITY_H

#include <string>
using std::string;
#include "FeatureVector.h"
#include "Label.h"

class Dataset;
class InputReader;
class Model;
class TrainingObjective;
class WeightVector;

class Utility {

  public:
  
    static bool loadDataset(const InputReader& reader, string fileName,
      Dataset& dataset);

    static void addRegularizationL2(const WeightVector& W, const double beta,
      double& fval, FeatureVector<RealWeight>& grad);
        
    static void evaluate(const WeightVector& w, TrainingObjective& obj,
      const Dataset& eval, const char* identifier, const string outFname = "");
        
    static double sigmoid(double a);
    
    static double log1Plus(double a);
    
    static double hinge(double a);
    
    // Return the zero-one loss, given the correct label y and guess yhat.
    static double delta(Label y, Label yhat);

  private:
  
    // Call the library function log1p if argument to log1Plus is less than
    // this value.
    static const double log1PlusTiny;
};

inline double Utility::sigmoid(double a) {
  return 1.0 / (1.0 + exp(-a));
}

inline double Utility::log1Plus(double a) {
  if (a <= log1PlusTiny)
    return (1.0F - 0.5*a)*a; // Source: http://goo.gl/7GgJz
  return log(1.0F + a);
}

inline double Utility::hinge(double a) {
  if (a > 0)
    return a;
  return 0;
}

inline double Utility::delta(Label y, Label yhat) {
  if (y == yhat)
    return 0;
  return 1;
}

#endif
