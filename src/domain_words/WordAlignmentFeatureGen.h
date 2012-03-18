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
#include "FeatureVector.h"
#include "Label.h"
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>
#include <set>
#include <string>
#include <vector>
using boost::regex;
using namespace std;

class Alphabet;
class Pattern;

class WordAlignmentFeatureGen : public AlignmentFeatureGen {
  public:

    WordAlignmentFeatureGen(boost::shared_ptr<Alphabet> alphabet, int order = 1,
      bool includeStateNgrams = true, bool includeAlignNgrams = true,
      bool includeCollapsedAlignNgrams = true, bool includeOpFeature = true,
      bool normalize = true);
      
    virtual ~WordAlignmentFeatureGen() {}
    
    //i: Current position in the source string.
    //j: Current position in the target string.
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
      Label label, int i, int j, const EditOperation& op,
      const vector<AlignmentPart>& editHistory);
      
    virtual int processOptions(int argc, char** argv);
    
    virtual double getDefaultFeatureWeight(const string& feature) const;
    
    static const string& name() {
      static const string _name = "WordAlignment";
      return _name;
    }

  private:
    void addFeatureId(const string& f, set<int>& featureIds) const;
    
    // Extracts a sequence of strings, starting with first and ending with
    // last-1, from the given string vector.
    static string extractPhrase(const vector<string>& str, int first, int last);
      
    int _order;
    
    bool _includeStateNgrams;
    
    bool _includeAlignNgrams;
    
    bool _includeCollapsedAlignNgrams;
    
    bool _includeOpFeature;
    
    bool _normalize;
    
    bool _regexEnabled;
    
    regex _regVowel;
  
    regex _regConsonant;
    
    regex _regPhraseSepMulti;
    
    regex _regPhraseSepLeadTrail;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    WordAlignmentFeatureGen(const WordAlignmentFeatureGen& x);
    WordAlignmentFeatureGen& operator=(const WordAlignmentFeatureGen& x);

};
#endif
