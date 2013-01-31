/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LOGLINEARMULTIELFV_H
#define _LOGLINEARMULTIELFV_H

#include "Parameters.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include <string>
#include <vector>

class Dataset;
class LogWeight;
class Model;

class LogLinearMultiELFV : public TrainingObjective {

  public:
  
    LogLinearMultiELFV(const Dataset& dataset, const std::vector<Model*>& models) :
      TrainingObjective(dataset, models) {}
    
    virtual ~LogLinearMultiELFV() {}
    
    virtual bool isBinary() const { return false; }
    
    virtual bool isUW() const { return true; }

    static const std::string& name() {
      static const std::string _name = "LogLinearMultiELFV";
      return _name;
    }
    
  private:
  
    virtual void valueAndGradientPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, RealVec& gradFv);
      
    virtual void predictPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
};

#endif
