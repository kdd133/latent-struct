/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _FEATUREGEN_H
#define _FEATUREGEN_H

#include "FeatureVector.h"
#include "FeatureVectorPool.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>
using namespace std;

class Alphabet;
class EditOperation;
class RealWeight;

class FeatureGen {
  
  public:
  
    FeatureGen(boost::shared_ptr<Alphabet> alphabet) :
      _alphabet(alphabet), _pool(0), _maxEntries(0) {}
      
    virtual ~FeatureGen() {}
    
    virtual int processOptions(int argc, char** argv) = 0;
    
    void enablePool(FeatureVectorPool<RealWeight>* pool) {
      _pool = pool;
    }
    
    void disablePool() {
      _pool = 0;
    }
    
    const FeatureVectorPool<RealWeight>* getPool() const {
      return _pool;
    }
    
    size_t getMaxEntries() const {
      return _maxEntries;
    }
    
    void setAlphabet(boost::shared_ptr<Alphabet> alphabet) {
      _alphabet = alphabet;
    }
    
    boost::shared_ptr<Alphabet> getAlphabet() const {
      return _alphabet;
    }
    
  protected:
  
    boost::shared_ptr<Alphabet> _alphabet;
  
    FeatureVectorPool<RealWeight>* _pool;
    
    size_t _maxEntries;
};

#endif
