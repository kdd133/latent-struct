/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _MAXMARGINBINARY_H
#define _MAXMARGINBINARY_H

#include "Parameters.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include <boost/ptr_container/ptr_vector.hpp>
#include <string>
#include <vector>

class Dataset;
class Model;

class MaxMarginBinary : public TrainingObjective {

  public:
  
    MaxMarginBinary(const Dataset& dataset, const std::vector<Model*>& models) :
      TrainingObjective(dataset, models) {}
    
    virtual ~MaxMarginBinary();

    virtual bool isBinary() const { return true; }

    static const std::string& name() {
      static const std::string _name = "MaxMarginBinary";
      return _name;
    }
    
  private:
  
    virtual void valueAndGradientPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, RealVec& gradFv);
      
    virtual void predictPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
      
    virtual void setLatentFeatureVectorsPart(const Parameters& theta, Model& model,
        const Dataset::iterator& begin, const Dataset::iterator& end);
        
    virtual void initLatentFeatureVectors(const Parameters& theta);
    
    virtual void clearLatentFeatureVectors();
  
    boost::ptr_vector<SparseRealVec> _imputedFvs;
};

#endif
