/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _MAXMARGINMULTIPIPELINEUW_H
#define _MAXMARGINMULTIPIPELINEUW_H

#include "Parameters.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>

class Dataset;
class LogWeight;
class Model;
class StringPairAligned;

class MaxMarginMultiPipelineUW : public TrainingObjective {

  public:
  
    MaxMarginMultiPipelineUW(const Dataset& dataset,
      const std::vector<Model*>& models) : TrainingObjective(dataset, models) {}

    virtual ~MaxMarginMultiPipelineUW() {}
    
    virtual bool isBinary() const { return false; }
    
    virtual bool isUW() const { return true; }

    static const std::string& name() {
      static const std::string _name = "MaxMarginMultiPipelineUW";
      return _name;
    }
    
  private:
  
    boost::unordered_map<std::pair<std::size_t, Label>,
      std::vector<StringPairAligned> > _kBestMap;

    boost::mutex _lock; // used to synchronize access to _kBestMap
      
    double bestAlignmentScore(const std::vector<StringPairAligned>& alignments,
        const WeightVector& weights, Model& model, const Label y);
    
    virtual void initializeKBestPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k);
  
    virtual void valueAndGradientPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, SparseRealVec& gradFv);
      
    virtual void predictPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
};

#endif
