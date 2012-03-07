/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _ALIGNMENTFEATUREGEN_H
#define _ALIGNMENTFEATUREGEN_H

#include "FeatureGen.h"
#include "FeatureVector.h"
#include "StateType.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>
using namespace std;

class EditOperation;
class Pattern;
class RealWeight;

class AlignmentFeatureGen : public FeatureGen {

  public:
  
    AlignmentFeatureGen(boost::shared_ptr<Alphabet> a) : FeatureGen(a) {}
    
    virtual ~AlignmentFeatureGen() {}
    
    //i: Current position in the source string.
    //j: Current position in the target string.
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
      int i, int j, int iNew, int jNew, int label,
      const EditOperation& op, const vector<StateType>& editHistory) = 0;
      
    virtual double getDefaultFeatureWeight(const string& feature) const = 0;
      
    virtual int processOptions(int argc, char** argv) = 0;
};

#endif
