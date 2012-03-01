/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _KLEMENTIEVROTHSENTENCEFEATUREGEN_H
#define _KLEMENTIEVROTHSENTENCEFEATUREGEN_H

#include "Alphabet.h"
#include "FeatureVector.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include <boost/shared_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include <vector>
using namespace std;

class Pattern;
class RealWeight;


class KlementievRothSentenceFeatureGen : public ObservedFeatureGen {

  public:
  
    KlementievRothSentenceFeatureGen(boost::shared_ptr<Alphabet> alphabet,
      bool normalize = true);
    
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
      const Label y);
        
    virtual int processOptions(int argc, char** argv);
    
    static const string& name() {
      static const string _name = "KRSentence";
      return _name;
    }

  private:
  
    int _substringSize;
    
    int _offsetSize;
    
    bool _normalize;
    
    bool _addBias;
    
    static void appendFeature(const int fi, const vector<string>* s,
        size_t i, size_t k, size_t end, vector<string>& subs,
        const boost::char_separator<char>& sep);
};

#endif
