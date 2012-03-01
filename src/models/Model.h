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
#include "FeatureVector.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include "WeightVector.h"
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;

class InputReader;
class LogWeight;
class Pattern;
class RealWeight;

class Model {

  public:
  
    Model(shared_ptr<AlignmentFeatureGen> fgenAlign,
      shared_ptr<ObservedFeatureGen> fgenObserved);
  
    virtual ~Model() {}
    
    virtual int processOptions(int argc, char** argv) = 0;

    // Returns the number of FeatureVector objects that were allocated.
    virtual size_t gatherFeatures(const Pattern& pattern,
      const Label label) = 0;
    
    virtual LogWeight totalMass(const WeightVector& w, const Pattern& pattern,
      const Label label) = 0;
        
    virtual RealWeight viterbiScore(const WeightVector& w,
      const Pattern& pattern, const Label label) = 0;

    virtual RealWeight maxFeatures(const WeightVector& w,
      FeatureVector<RealWeight>& fv, const Pattern& pattern, const Label label,
      bool includeObservedFeaturesArc = true) = 0;
    
    // Returns total mass for this Pattern and Label.
    virtual LogWeight expectedFeatures(const WeightVector& w,
      FeatureVector<LogWeight>& fv, const Pattern& pattern, const Label label,
      bool normalize = true) = 0;
      
    // Returns true of the caller assumes ownership of the FeatureVector.
    virtual FeatureVector<RealWeight>* observedFeatures(const Pattern& pattern,
      const Label label, bool& callerOwns) = 0;
    
    virtual void emptyCache() = 0;
    
    void setCacheEnabled(bool state);
    
    bool getCacheEnabled() const;
    
    shared_ptr<ObservedFeatureGen> getFgenObserved() const;
    
    shared_ptr<AlignmentFeatureGen> getFgenLatent() const;
    

  protected:
  
    // The alignment feature generator.
    shared_ptr<AlignmentFeatureGen> _fgenAlign;
    
    // The observed feature generator.
    shared_ptr<ObservedFeatureGen> _fgenObserved;
    
    // Whether or not to cache transducers in memory during training.
    bool _cacheFsts;
    
    
  private:
  
    Model& operator=(const Model& model);
    
    Model(const Model& model);
};

inline Model::Model(shared_ptr<AlignmentFeatureGen> fgenAlign,
    shared_ptr<ObservedFeatureGen> fgenObserved) :
    _fgenAlign(fgenAlign), _fgenObserved(fgenObserved), _cacheFsts(false) {
  assert(_fgenAlign != 0);
  assert(_fgenObserved != 0);
}

inline void Model::setCacheEnabled(bool state) {
  _cacheFsts = state;
}

inline bool Model::getCacheEnabled() const {
  return _cacheFsts;
}

inline shared_ptr<ObservedFeatureGen> Model::getFgenObserved() const {
  return _fgenObserved;
}

inline shared_ptr<AlignmentFeatureGen> Model::getFgenLatent() const {
  return _fgenAlign;
}

#endif
