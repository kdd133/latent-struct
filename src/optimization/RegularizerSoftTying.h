/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _REGULARIZERSOFTTYING_H
#define _REGULARIZERSOFTTYING_H

#include "Alphabet.h"
#include "Label.h"
#include "Parameters.h"
#include "Regularizer.h"
#include "Ublas.h"

// The soft-tying regularizer adds the following penalty to the objective:   
//     \sum_{k=1}^{K}\frac{\beta}{2}||w^k-w^0||_2^2
// where K is the number of classes, w^k are the weights in the vector w
// that correspond to the kth class, and w^0 is a set of shared reference
// parameters. The idea behind this scheme is that the shared parameters act as
// a prior (in a Bayesian sense) on the weights w^k for a given class k. If a
// class has little evidence for a particular feature, the prior will
// effectively override the value of the weight that is learned. On the other
// hand, if a class has strong evidence for a feature weight, this will
// override the prior. These concepts are described in more detail in the paper
// "Hierarchical Bayesian Domain Adaptation" by Finkel and Manning (2009).
class RegularizerSoftTying : public Regularizer {

  public:

    RegularizerSoftTying(double beta = 1e-4);

    virtual ~RegularizerSoftTying() {}

    virtual void addRegularization(const Parameters& theta, double& fval,
        RealVec& grad) const;
        
    virtual void setupParameters(Parameters& theta, Alphabet& alphabet,
        const std::set<Label>& labelSet);
        
    virtual void setBeta(double beta);
        
    static const std::string& name() {
      static const std::string _name = "SoftTying";
      return _name;
    }
    
    virtual int processOptions(int argc, char** argv);
    
  private:
  
    // Note: The _beta value (from the parent class) will be ignored, other
    // than being used to initialize these variables in the constructor.
    double _betaW;
    double _betaSharedW;
    double _betaU;
    double _betaSharedU;
  
    Alphabet* _alphabet;
  
    const std::set<Label>* _labels;
};

#endif
