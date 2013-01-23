/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _WORDALIGNMENTFEATUREGEN_H
#define _WORDALIGNMENTFEATUREGEN_H


#include "AlignmentFeatureGen.h"
#include "AlignmentPart.h"
#include "Label.h"
#include "Ublas.h"
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>
#include <set>
#include <string>
#include <vector>

class Alphabet;
class Pattern;

class WordAlignmentFeatureGen : public AlignmentFeatureGen {
  public:

    WordAlignmentFeatureGen(boost::shared_ptr<Alphabet> alphabet);
      
    virtual ~WordAlignmentFeatureGen() {}
    
    //i: Current position in the source string.
    //j: Current position in the target string.
    virtual SparseRealVec* getFeatures(const Pattern& x, Label label, int i,
      int j, const EditOperation& op,
      const std::vector<AlignmentPart>& editHistory);
      
    virtual int processOptions(int argc, char** argv);
    
    virtual double getDefaultFeatureWeight(const std::string& feature,
      Label label) const;
    
    static const std::string& name() {
      static const std::string _name = "WordAlignment";
      return _name;
    }

  private:
    void addFeatureId(const std::string& f, Label y, std::set<int>& featureIds)
      const;
    
    // Extracts a sequence of strings, starting with first and ending with
    // last-1, from the given string vector.
    static std::string extractPhrase(const std::vector<std::string>& str,
      int first, int last);
      
    int _order;
    
    bool _includeStateNgrams;
    
    bool _includeAlignNgrams;
    
    bool _includeCollapsedAlignNgrams;
    
    bool _includeBigramFeatures;
    
    bool _normalize;
    
    bool _regexEnabled;
    
    bool _alignUnigramsOnly;
    
    bool _stateUnigramsOnly;
    
    boost::regex _regVowel;
  
    boost::regex _regConsonant;
    
    boost::regex _regPhraseSepMulti;
    
    boost::regex _regPhraseSepLeadTrail;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    WordAlignmentFeatureGen(const WordAlignmentFeatureGen& x);
    WordAlignmentFeatureGen& operator=(const WordAlignmentFeatureGen& x);

};
#endif
