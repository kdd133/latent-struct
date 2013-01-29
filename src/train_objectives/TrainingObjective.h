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
#include "Label.h"
#include "LabelScoreTable.h"
#include "Model.h"
#include "Parameters.h"
#include "Ublas.h"
#include <boost/ptr_container/ptr_vector.hpp>
#include <vector>

class Pattern;

class TrainingObjective {

  public:

    TrainingObjective(const Dataset& dataset,
        const std::vector<Model*>& models);
    
    virtual ~TrainingObjective() {}
  
    virtual void valueAndGradient(const Parameters& theta, double& value,
      RealVec& grad);
    
    virtual void predict(const Parameters& theta, const Dataset& evalData,
      LabelScoreTable& scores);
    
    // Called once, prior to training. Typically used to allocate on or more
    // FeatureVector objects.
    virtual void initLatentFeatureVectors(const Parameters& theta);
    
    // Assigns each thread a partition of the data, and instructs it to
    // compute the latent feature vectors for the given partition, via
    // setLatentFeatureVectorsPart.
    virtual void setLatentFeatureVectors(const Parameters& theta);
    
    virtual bool isBinary() const = 0;
    
    // Does this objective learn separate parameters for latent variable
    // imputation and classification?
    virtual bool isUW() const {
      return false; // false by default
    }
    
    // The purpose of this method is to let the caller know how many parameters
    // this objective will learn, based on the given number of features.
    virtual Parameters getDefaultParameters(std::size_t numFeatures) const;
    
    void setComputeAverageLoss(bool state) {
      _computeAverageLoss = state;
    }
    
    Model& getModel(std::size_t modelNum) {
      return _models[modelNum];
    }
    
    std::size_t getNumModels() const {
      return _models.size();
    }
    
    void gatherFeatures(std::size_t& maxFvs, std::size_t& totalFvs);
    
    void combineAndLockAlphabets(const std::set<Label>& labels);
    
    static const Label kPositive;
    
  protected:
  
    const Dataset& _dataset;
  
    // A vector of pointers to Model objects, of which this class assumes
    // ownership. Each model will be used in a separate thread of execution.
    boost::ptr_vector<Model> _models;
    
    // If true, divide the loss by 1/t, where t is the number of examples.
    bool _computeAverageLoss;
    
    virtual void valueAndGradientPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label maxLabel, double& f, RealVec& g) = 0;
    
    // After valueAndGradientPart() completes for each thread, this function is
    // called and may modify the function value and/or gradient. For example,
    // MaxMarginMulti requires us to account for the imputed feature vector
    // exactly once -- but not once per thread.
    virtual void valueAndGradientFinalize(const Parameters& theta, double& f,
        RealVec& g);
      
    virtual void predictPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label maxLabel, LabelScoreTable& scores) = 0;
      
    // Called by setLatentFeatureVectors prior to delegating computations to
    // setLatentFeatureVectorsPart. For example, used to zero out the feature
    // vectors that were computed on the previous iteration.
    virtual void clearLatentFeatureVectors();
      
    virtual void setLatentFeatureVectorsPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end);
      
    virtual void gatherFeaturesPart(Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label maxLabel, std::size_t& maxFvs, std::size_t& totalFvs);
};

#endif
