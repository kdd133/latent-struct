/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _KLEMENTIEVROTHWORDFEATUREGEN_H
#define _KLEMENTIEVROTHWORDFEATUREGEN_H

#include "Alphabet.h"
#include "FeatureVector.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include <boost/regex.hpp>
#include <string>
#include <vector>
using boost::regex;
using namespace std;

class Pattern;
class RealWeight;


class KlementievRothWordFeatureGen : public ObservedFeatureGen {

  public:
  
    KlementievRothWordFeatureGen(boost::shared_ptr<Alphabet> alphabet,
      bool normalize = true);
    
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
      const Label y);
        
    virtual int processOptions(int argc, char** argv);
    
    static const string& name() {
      static const string _name = "KRWord";
      return _name;
    }

    static const string CHAR_JOINER;
    static const string SUB_JOINER;

  private:
  
    int _substringSize;
    
    int _offsetSize;
    
    bool _normalize;
    
    bool _addBias;
    
    bool _regexEnabled;
    
    regex _regVowel;
  
    regex _regConsonant;
    
    static void appendSubstrings(const vector<string>* s, size_t i, size_t k,
      size_t end, vector<string>& subs);
};

#endif
