/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LOGLINEARBINARYUNSCALED_H
#define _LOGLINEARBINARYUNSCALED_H

#include "FeatureVector.h"
#include "TrainingObjective.h"

class Dataset;
class Model;
class RealWeight;

class LogLinearBinaryUnscaled : public TrainingObjective {

  public:
  
    LogLinearBinaryUnscaled(const Dataset& dataset,
      const vector<Model*>& models) : TrainingObjective(dataset, models) {}
    
    virtual ~LogLinearBinaryUnscaled() {}
    
    virtual bool isBinary() const { return true; }

    static const string& name() {
      static const string _name = "LogLinearBinaryUnscaled";
      return _name;
    }
    
  private:
  
    virtual void valueAndGradientPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, FeatureVector<RealWeight>& gradFv);
      
    virtual void predictPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
};

#endif
