/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _MAXMARGINBINARYPIPELINEUW_H
#define _MAXMARGINBINARYPIPELINEUW_H

#include "Parameters.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include <string>
#include <vector>

class Dataset;
class LogWeight;
class Model;

class MaxMarginBinaryPipelineUW : public TrainingObjective {

  public:
  
    MaxMarginBinaryPipelineUW(const Dataset& dataset,
      const std::vector<Model*>& models) :
      TrainingObjective(dataset, models) {}
    
    virtual ~MaxMarginBinaryPipelineUW() {}
    
    virtual bool isBinary() const { return true; }
    
    virtual bool isUW() const { return true; }

    static const std::string& name() {
      static const std::string _name = "MaxMarginBinaryPipelineUW";
      return _name;
    }
    
  private:
  
    virtual void valueAndGradientPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, SparseRealVec& gradFv);
      
    virtual void predictPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
};

#endif
