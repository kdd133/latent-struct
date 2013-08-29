/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _BERGSMAKONDRAKWORDFEATUREGEN_H
#define _BERGSMAKONDRAKWORDFEATUREGEN_H

#include "Alphabet.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include "Ublas.h"
#include <boost/unordered_map.hpp>
#include <deque>
#include <string>
#include <vector>

class Pattern;


class BergsmaKondrakWordFeatureGen : public ObservedFeatureGen {

  public:
  
    BergsmaKondrakWordFeatureGen(boost::shared_ptr<Alphabet> alphabet,
      bool normalize = true);
    
    virtual SparseRealVec* getFeatures(const Pattern& x, const Label y);
        
    virtual int processOptions(int argc, char** argv);
    
    static const std::string& name() {
      static const std::string _name = "BKWord";
      return _name;
    }

    static const std::string CHAR_JOINER;
    static const std::string SUB_JOINER;
    static const std::string MISMATCH_PREFIX;

  private:
  
    int _substringSize;
    
    bool _normalize;
    
    bool _addMismatches;
    
    bool _collapseMismatches;
    
    bool _addBias;
      
    void getPhrasePairs(const std::vector<std::string>& s,
        const std::vector<std::string>& t, int sk, int tk,
        boost::unordered_map<int, int>& fv, const bool* match, const Label y);
      
    void appendPhrasePair(const std::vector<std::string>& s,
        const std::vector<std::string>& t, const std::deque<int>& sw,
        const std::deque<int>& tw, boost::unordered_map<int, int>& fv,
        const Label y);
        
    void getMismatches(const std::vector<std::string>& s,
        const std::vector<std::string>& t, boost::unordered_map<int, int>& fv,
        const Label y);
};

#endif
