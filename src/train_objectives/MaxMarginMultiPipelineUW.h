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
#include <boost/scoped_ptr.hpp>
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
    
    virtual void clearKBest();

    static const std::string& name() {
      static const std::string _name = "MaxMarginMultiPipelineUW";
      return _name;
    }
    
    typedef struct {
      boost::shared_ptr<std::vector<StringPairAligned> > alignments;
      boost::shared_ptr<std::vector<boost::shared_ptr<SparseRealVec> > > maxFvs;
    } KBestInfo;
    
  private:
  
    boost::unordered_map<std::pair<std::size_t, Label>, KBestInfo> _kBestMap;

    boost::mutex _lock; // used to synchronize access to _kBestMap
      
    // Returns the score of the highest-scoring alignment, and its index in the
    // alignments vector.
    double bestAlignment(const std::vector<StringPairAligned>& alignments,
        const WeightVector& weights, Model& model, const Label y,
        int* indexBest = 0);
    
    virtual void initKBestPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k);
  
    virtual void valueAndGradientPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, double& funcVal, SparseRealVec& gradFv);
      
    virtual void predictPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label k, LabelScoreTable& scores);
      
    virtual void gatherFeaturesPart(Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end,
      const Label maxLabel, std::size_t& maxFvs, std::size_t& totalFvs);
    
    virtual void setLatentFeatureVectorsPart(const Parameters& theta, Model& model,
      const Dataset::iterator& begin, const Dataset::iterator& end);
    
    virtual void initLatentFeatureVectors(const Parameters& theta);
    
    virtual void clearLatentFeatureVectors();
    
    virtual void valueAndGradientFinalize(const Parameters& theta, double& f,
      SparseRealVec& g);
      
    const KBestInfo& fetchKBestInfo(const Pattern& x, Label y);
  
    boost::scoped_ptr<RealVec> _imputedFv;
    
    boost::mutex _flag; // used to synchronize access to _imputedFv
};

#endif
