/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _MODEL_H
#define _MODEL_H

#include "AlignmentFeatureGen.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include "Ublas.h"
#include "WeightVector.h"
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <vector>

class InputReader;
class LogWeight;
class Pattern;

class Model {

  public:
  
    Model(boost::shared_ptr<AlignmentFeatureGen> fgenAlign,
      boost::shared_ptr<ObservedFeatureGen> fgenObserved);
  
    virtual ~Model() {}
    
    virtual int processOptions(int argc, char** argv) = 0;

    // Returns the number of FeatureVector objects that were allocated.
    virtual size_t gatherFeatures(const Pattern& pattern,
      const Label label) = 0;
    
    virtual LogWeight totalMass(const WeightVector& w, const Pattern& pattern,
      const Label label) = 0;
        
    virtual double viterbiScore(const WeightVector& w, const Pattern& pattern,
      const Label label) = 0;

    virtual double maxFeatures(const WeightVector& w, SparseRealVec* fv,
      const Pattern& pattern, const Label label,
      bool includeObservedFeaturesArc = true) = 0;
    
    // Returns total mass for this Pattern and Label.
    virtual LogWeight expectedFeatures(const WeightVector& w, SparseLogVec* fv,
      const Pattern& pattern, const Label label, bool normalize = true) = 0;
      
    // Returns a lower triangular matrix representing the symmetric matrix of
    // feature cooccurrences.
    virtual LogWeight expectedFeatureCooccurrences(
      const WeightVector& w, AccumLogMat* fm, SparseLogVec* fv,
      const Pattern& pattern, const Label label, bool normalize = true) = 0;
      
    virtual boost::shared_ptr<const SparseRealVec> observedFeatures(
      const Pattern& pattern, const Label label) = 0;
      
    virtual void getBestAlignments(std::ostream& alignmentStringRepresentations,
      boost::shared_ptr<std::vector<boost::shared_ptr<const SparseRealVec> > >& maxFvs,
      const WeightVector& w, const Pattern& pattern, const Label label,
      bool includeObservedFeatures = true) = 0;
    
    virtual void emptyCache() = 0;
    
    void setCacheEnabled(bool state);
    
    bool getCacheEnabled() const;
    
    boost::shared_ptr<ObservedFeatureGen> getFgenObserved() const;
    
    boost::shared_ptr<AlignmentFeatureGen> getFgenLatent() const;
    
    void onlyCacheIdsGreaterThanOrEqualTo(std::size_t id);
    

  protected:
  
    // The alignment feature generator.
    boost::shared_ptr<AlignmentFeatureGen> _fgenAlign;
    
    // The observed feature generator.
    boost::shared_ptr<ObservedFeatureGen> _fgenObserved;
    
    // Whether or not to cache graphs in memory during training.
    bool _cacheGraphs;
    
    // Do not cache a graph for an example that has an id less than or equal
    // to this value.
    std::size_t _onlyCacheIdsGreaterThanOrEqualTo;
    
  private:
  
    Model& operator=(const Model& model);
    
    Model(const Model& model);
};

inline Model::Model(boost::shared_ptr<AlignmentFeatureGen> fgenAlign,
    boost::shared_ptr<ObservedFeatureGen> fgenObserved) :
    _fgenAlign(fgenAlign), _fgenObserved(fgenObserved), _cacheGraphs(false),
    _onlyCacheIdsGreaterThanOrEqualTo(0) {
  assert(_fgenAlign != 0);
  assert(_fgenObserved != 0);
}

inline void Model::setCacheEnabled(bool state) {
  _cacheGraphs = state;
}

inline bool Model::getCacheEnabled() const {
  return _cacheGraphs;
}

inline boost::shared_ptr<ObservedFeatureGen> Model::getFgenObserved() const {
  return _fgenObserved;
}

inline boost::shared_ptr<AlignmentFeatureGen> Model::getFgenLatent() const {
  return _fgenAlign;
}

inline void Model::onlyCacheIdsGreaterThanOrEqualTo(std::size_t id) {
  _onlyCacheIdsGreaterThanOrEqualTo = id;
}

#endif
