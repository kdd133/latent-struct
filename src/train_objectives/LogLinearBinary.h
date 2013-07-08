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

#include "Parameters.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>

class Dataset;
class LogWeight;
class Model;

class LogLinearBinary : public TrainingObjective {

  public:
  
    typedef boost::unordered_map<int,LogWeight> DictType;
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
    
    virtual void valueAndGradientPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, SparseRealVec& gradFv);
      
    virtual void predictPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
};

#endif
