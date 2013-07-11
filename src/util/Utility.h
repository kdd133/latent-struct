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

#include "Label.h"
#include "LabelScoreTable.h"
#include "Ublas.h"
#include <boost/shared_array.hpp>
#include <string>
#include <vector>

class Dataset;
class InputReader;
class Model;
class Parameters;
class TrainingObjective;

class Utility {

  public:
  
    static bool loadDataset(const InputReader& reader, std::string fileName,
      Dataset& dataset, std::size_t firstId = 0);

    static void evaluate(const Parameters& w, TrainingObjective& obj,
      const Dataset& eval, const std::string& identifier,
      const std::string& outFname = "");
      
    static void evaluate(const std::vector<Parameters>& weightVectors,
      TrainingObjective& obj, const Dataset& evalData,
      const std::vector<std::string>& ids,
      const std::vector<std::string>& fnames, bool enableCache);
      
    static void calcPerformanceMeasures(const Dataset& evalData,
      LabelScoreTable& scores, bool printToStdout,
      const std::string& identifier, const std::string& fnameOut,
      double& accuracy, double& precision, double& recall, double& fscore,
      double& avg11ptPrec);
        
    static double sigmoid(double a);
    
    static double log1Plus(double a);
    
    static double hinge(double a);
    
    // Return the zero-one loss, given the correct label y and guess yhat.
    static double delta(Label y, Label yhat);
    
    static boost::shared_array<double> generateGaussianSamples(std::size_t n,
      double mean, double stdev, int seed = 0); 

    // Return a random permutation of the integers 0,...,n-1.
    static boost::shared_array<int> randPerm(int n, int seed = 0);
    
    // Returns the ith coordinate of the numerical gradient (i.e., computed via
    // finite differences) for the given objective value. Specifically, it
    // computes (f(theta + eps_i) - f(theta - eps_i)) / (2 * eps), where eps_i
    // is a vector v of zeros except for v[i]=eps. See this URL for details:
    // http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_
    // and_advanced_optimization
    static double getNumericalGradientForCoordinate(TrainingObjective& obj,
      const Parameters& theta, int i, double epsilon = 1e-4);
      
    // Returns the Levenshtein edit distance between the source and target
    // strings, which are represented as vectors of strings. The sourceEps
    // and targetEps arguments will be set to the aligned source and target
    // strings, respectively, with "-" symbols indicating insertions/deletions. 
    static int levenshtein(const std::vector<std::string>& source,
                           const std::vector<std::string>& target,
                           std::vector<std::string>& sourceEps,
                           std::vector<std::string>& targetEps,
                           int subCost = 1);
                           
  private:
  
    // Call the library function log1p if argument to log1Plus is less than
    // this value.
    static const double log1PlusTiny;
    
    struct prediction {
      Label y;
      double positiveScore;
      bool operator<(const prediction& p) const {
        return positiveScore < p.positiveScore;
      }
    };
    
    // Note: The predictions vector will be modified (sorted).
    static double avg11ptPrecision(std::vector<prediction>& predictions);
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
