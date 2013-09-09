/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _BERGSMAKONDRAKPHRASEPAIRS_H
#define _BERGSMAKONDRAKPHRASEPAIRS_H

#include "Alphabet.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include "Ublas.h"
#include <boost/unordered_map.hpp>
#include <deque>
#include <string>
#include <vector>

class Pattern;


class BergsmaKondrakPhrasePairs : public ObservedFeatureGen {

  public:
  
    BergsmaKondrakPhrasePairs(boost::shared_ptr<Alphabet> alphabet,
      bool normalize = true);
    
    virtual SparseRealVec* getFeatures(const Pattern& x, const Label y);
        
    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "BKPhrasePairs";
      return _name;
    }

  private:
  
    bool _normalize;
    
    bool _addMismatches;
    
    bool _addBias;
    
    bool _addNed;
};

#endif
