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
#include "Ublas.h"
#include <boost/shared_ptr.hpp>
#include <set>
#include <string>
#include <vector>

class Alphabet;
class Pattern;
class StringPair;

class SentenceAlignmentFeatureGen : public AlignmentFeatureGen {
  public:

    SentenceAlignmentFeatureGen(boost::shared_ptr<Alphabet> alphabet);
      
    virtual ~SentenceAlignmentFeatureGen() {}
    
    //i: Current position in the source string.
    //j: Current position in the target string.
    virtual SparseRealVec* getFeatures(const Pattern& x, Label label, int i,
      int j, const EditOperation& op,
      const std::vector<AlignmentPart>& editHistory);
      
    virtual int processOptions(int argc, char** argv);
    
    virtual double getDefaultFeatureWeight(const std::string& feature) const;
    
    static const std::string& name() {
      static const std::string _name = "SentenceAlignment";
      return _name;
    }
      
  private:
  
    void addFeatureId(const std::string& f, std::set<int>& featureIds) const;
      
    int _order;
    
    bool _includeStateNgrams;
    
    bool _includeAlignNgrams;
    
    bool _alignUnigramsOnly;
    
    bool _normalize;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    SentenceAlignmentFeatureGen(const SentenceAlignmentFeatureGen& x);
    SentenceAlignmentFeatureGen& operator=(const SentenceAlignmentFeatureGen& x);

};
#endif
