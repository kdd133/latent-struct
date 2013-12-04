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
#include <boost/ptr_container/ptr_map.hpp>
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
    
    // A data structure that represents a k-best list.
    typedef struct {
      // A string representation of the k-best alignments.
      // See, e.g., StringEditModel::getBestAlignments().
      std::string alignStrings;
      
      // The latent feature vectors (based on parameters u) that correspond to
      // the k-best alignments.
      boost::shared_ptr<std::vector<boost::shared_ptr<
        const SparseRealVec> > > maxFvs;
        
      // The observed feature vectors (based on parameters w) that correspond to
      // the k-best alignments.
      boost::shared_ptr<std::vector<boost::shared_ptr<
        const SparseRealVec> > > observedFvs;
    } KBestInfo;
    
  private:
  
    boost::ptr_map<std::pair<std::size_t, Label>, KBestInfo> _kBestMap;

    boost::mutex _lock; // used to synchronize access to _kBestMap
      
    // Returns the score of the highest-scoring alignment, and its index in the
    // alignments vector.
    double bestAlignmentScore(const KBestInfo& kBest,
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
      
    KBestInfo* fetchKBestInfo(const Pattern& x, Label y);
  
    boost::scoped_ptr<RealVec> _imputedFv;
    
    boost::mutex _flag; // used to synchronize access to _imputedFv
    
    void maxZ(const KBestInfo& kBest, const Label y, const Parameters& theta,
        Model& model, double& scoreMaxUW, int& indexMaxUW, double& scoreMaxU,
        int& indexMaxU);
};

#endif
