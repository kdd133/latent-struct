/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _TRAININGOBJECTIVE_H
#define _TRAININGOBJECTIVE_H

#include "Dataset.h"
#include "FeatureVector.h"
#include "Label.h"
#include "LabelScoreTable.h"
#include "Model.h"
#include <boost/ptr_container/ptr_vector.hpp>
#include <vector>
using std::vector;

class Pattern;
class RealWeight;
class WeightVector;

class TrainingObjective {

  public:

    TrainingObjective(const Dataset& dataset, const vector<Model*>& models);
    
    virtual ~TrainingObjective() {}
  
    virtual void valueAndGradient(const WeightVector& w, double& value,
      FeatureVector<RealWeight>& grad);
    
    virtual void predict(const WeightVector& w, const Dataset& evalData,
      LabelScoreTable& scores);
    
    virtual void initLatentFeatureVectors(const WeightVector& w);
    
    virtual void setLatentFeatureVectors(const WeightVector& w);
    
    virtual bool isBinary() const = 0;
    
    void setComputeAverageLoss(bool state) {
      _computeAverageLoss = state;
    }
    
    Model& getModel(size_t modelNum) {
      return _models[modelNum];
    }
    
    size_t getNumModels() const {
      return _models.size();
    }
    
    void gatherFeatures(size_t& maxFvs, size_t& totalFvs);
    
    void combineAndLockAlphabets();
    
    static const Label kPositive;
    
  protected:
  
    const Dataset& _dataset;
  
    // A vector of pointers to Model objects, of which this class assumes
    // ownership. Each model will be used in a separate thread of execution.
    boost::ptr_vector<Model> _models;
    
    // If true, divide the loss by 1/t, where t is the number of examples.
    bool _computeAverageLoss;
    
    virtual void valueAndGradientPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label maxLabel, double& f, FeatureVector<RealWeight>& g) = 0;
    
    // After valueAndGradientPart() completes for each thread, this function is
    // called and may modify the function value and/or gradient. For example,
    // MaxMarginMulti requires us to account for the imputed feature vector
    // exactly once -- but not once per thread.
    virtual void valueAndGradientFinalize(const WeightVector& w, double& f,
        FeatureVector<RealWeight>& g);
      
    virtual void predictPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label maxLabel, LabelScoreTable& scores) = 0;
      
    virtual void setLatentFeatureVectorsPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end);
      
    virtual void gatherFeaturesPart(Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label maxLabel, size_t& maxFvs, size_t& totalFvs);
};

#endif
