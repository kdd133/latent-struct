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
#include "FeatureVector.h"
#include "StateType.h"
#include <boost/shared_ptr.hpp>
#include <set>
#include <string>
#include <vector>
using namespace std;

class Alphabet;
class Pattern;
class StringPair;

class WordAlignmentFeatureGen : public AlignmentFeatureGen {
  public:

    WordAlignmentFeatureGen(boost::shared_ptr<Alphabet> alphabet,
      int order = 1, bool includeAnnotatedEdits = true,
      bool includeEditFeats = true, bool includeStateFeats = true,
      bool normalize = true);
      
    virtual ~WordAlignmentFeatureGen() {}
    
    //i: Current position in the source string.
    //j: Current position in the target string.
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
      int i, int j, int iNew, int jNew, 
      int label, const EditOperation& op, const vector<StateType>& editHistory);
      
    virtual int processOptions(int argc, char** argv);
    
    virtual double getDefaultFeatureWeight(const string& feature) const;
    
    static const string& name() {
      static const string _name = "WordAlignment";
      return _name;
    }
      
  private:
    void addFeatureId(const string& f, list<int>& featureIds) const;
    
    // Extracts a sequence of strings, starting with first and ending with
    // last-1, from the given string vector.
    static string extractPhrase(const vector<string>& str, int first, int last);
      
    int _order; // TODO: Change to size_t (make sure it doesn't break anything)
    
    bool _includeAnnotatedEdits;
    
    bool _includeEditFeats;
    
    bool _includeStateNgrams;
    
    bool _includeAlignNgrams;
    
    bool _normalize;
    
    bool _addContextFeats;
    
    // Handle matching and mismatching phrases differently, as in old code.
    bool _legacy;
    
    set<string> _vowels;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    WordAlignmentFeatureGen(const WordAlignmentFeatureGen& x);
    WordAlignmentFeatureGen& operator=(const WordAlignmentFeatureGen& x);

};
#endif
