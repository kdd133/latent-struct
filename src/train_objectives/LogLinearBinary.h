/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LOGLINEARBINARY_H
#define _LOGLINEARBINARY_H

#include "TrainingObjective.h"
#include "Ublas.h"
#include <string>
#include <tr1/unordered_map>
#include <vector>

class Dataset;
class LogWeight;
class Model;

class LogLinearBinary : public TrainingObjective {

  public:
  
    typedef std::tr1::unordered_map<int,LogWeight> DictType;
    typedef DictType::value_type PairType;
  
    LogLinearBinary(const Dataset& dataset, const std::vector<Model*>& models) :
      TrainingObjective(dataset, models) {}
    
    virtual ~LogLinearBinary() {}
    
    virtual bool isBinary() const { return true; }

    static const std::string& name() {
      static const std::string _name = "LogLinearBinary";
      return _name;
    }
    
  private:
    
    DictType _logSizeZxMap;
    
    virtual void valueAndGradientPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, RealVec& gradFv);
      
    virtual void predictPart(const WeightVector& w, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
};

#endif
