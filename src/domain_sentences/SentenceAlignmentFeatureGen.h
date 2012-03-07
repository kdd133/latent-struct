/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _SENTENCEALIGNMENTFEATUREGEN_H
#define _SENTENCEALIGNMENTFEATUREGEN_H

#include "AlignmentFeatureGen.h"
#include "AlignmentPart.h"
#include "FeatureVector.h"
#include "StateType.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>
using namespace std;

class Alphabet;
class Pattern;
class StringPair;

class SentenceAlignmentFeatureGen : public AlignmentFeatureGen {
  public:

    SentenceAlignmentFeatureGen(boost::shared_ptr<Alphabet> alphabet,
      int order = 1, bool includeAnnotatedEdits = true,
      bool includeEditFeats = true, bool includeStateFeats = true,
      bool normalize = true);
      
    virtual ~SentenceAlignmentFeatureGen() {}
    
    //i: Current position in the source string.
    //j: Current position in the target string.
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
      Label label, int i, int j, const EditOperation& op,
      const vector<AlignmentPart>& editHistory);
      
    virtual int processOptions(int argc, char** argv);
    
    virtual double getDefaultFeatureWeight(const string& feature) const;
    
    static const string& name() {
      static const string _name = "SentenceAlignment";
      return _name;
    }
    
    static bool getPhraseIterators(const vector<string>& str,
        int first, int last,
        vector<string>::const_iterator& itBegin,
        vector<string>::const_iterator& itEnd);
      
  private:
  
    void addFeatureId(const string& f, list<int>& featureIds) const;
      
    int _order;
    
    bool _includeAnnotatedEdits;
    
    bool _includeEditFeats;
    
    bool _includeStateNgrams;
    
    bool _normalize;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    SentenceAlignmentFeatureGen(const SentenceAlignmentFeatureGen& x);
    SentenceAlignmentFeatureGen& operator=(const SentenceAlignmentFeatureGen& x);

};
#endif
