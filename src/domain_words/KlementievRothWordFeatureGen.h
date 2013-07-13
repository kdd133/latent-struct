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
#include "Label.h"
#include "ObservedFeatureGen.h"
#include "Ublas.h"
#include <boost/regex.hpp>
#include <string>
#include <vector>

class Pattern;


class KlementievRothWordFeatureGen : public ObservedFeatureGen {

  public:
  
    KlementievRothWordFeatureGen(boost::shared_ptr<Alphabet> alphabet,
      bool normalize = true);
    
    virtual SparseRealVec* getFeatures(const Pattern& x, const Label y);
        
    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "KRWord";
      return _name;
    }

    static const std::string CHAR_JOINER;
    static const std::string SUB_JOINER;

  private:
  
    int _substringSize;
    
    int _offsetSize;
    
    bool _normalize;
    
    bool _addBias;
    
    bool _regexEnabled;
    
    bool _encodeOffset;
    
    bool _ignoreEps;
    
    boost::regex _regVowel;
  
    boost::regex _regConsonant;
    
    static void appendSubstrings(const std::vector<std::string>* s, std::size_t i,
      std::size_t k, std::size_t end, std::vector<std::string>& subs,
      bool ignoreEps = false, const std::string suffix = std::string());
};

#endif
