/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _MAXMARGINOBJECTIVE_H
#define _MAXMARGINOBJECTIVE_H

#include "TrainingObjective.h"
#include "Ublas.h"
#include <boost/scoped_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <string>
#include <vector>

class Dataset;
class Model;

class MaxMarginMulti : public TrainingObjective {

  public:
  
    MaxMarginMulti(const Dataset& dataset, const std::vector<Model*>& models) :
      TrainingObjective(dataset, models), _imputedFv(0) {}
    
    virtual ~MaxMarginMulti() {}

    virtual bool isBinary() const { return false; }

    static const std::string& name() {
      static const std::string _name = "MaxMarginMulti";
      return _name;
    }
    
  private:
  
    virtual void valueAndGradientPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, RealVec& gradFv);
      
    virtual void valueAndGradientFinalize(const WeightVector& w, double& f,
      RealVec& g);
      
    virtual void predictPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
      
    virtual void setLatentFeatureVectorsPart(const WeightVector& w, Model& model,
        const Dataset::iterator& begin, const Dataset::iterator& end);
        
    virtual void initLatentFeatureVectors(const WeightVector& w);
    
    virtual void clearLatentFeatureVectors();
  
    boost::scoped_ptr<SparseRealVec> _imputedFv;
    
    boost::mutex _flag;
};

#endif
