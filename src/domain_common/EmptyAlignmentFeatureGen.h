/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _EMPTYALIGNMENTFEATUREGEN_H
#define _EMPTYALIGNMENTFEATUREGEN_H

#include "AlignmentFeatureGen.h"
#include "AlignmentPart.h"
#include "FeatureVector.h"
#include "Label.h"
#include "StateType.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>
using namespace std;

class EditOperation;
class Pattern;
class RealWeight;

class EmptyAlignmentFeatureGen : public AlignmentFeatureGen {

  public:
  
    EmptyAlignmentFeatureGen(boost::shared_ptr<Alphabet> alphabet) :
      AlignmentFeatureGen(alphabet) {}
    
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
        Label label, int i, int j, const EditOperation& op,
        const vector<AlignmentPart>& editHistory) {
      return 0;
    }
      
    virtual ~EmptyAlignmentFeatureGen() {}
    
    virtual int processOptions(int argc, char** argv) {
      return 0;
    }
    
    virtual double getDefaultFeatureWeight(const string& feature) const {
      return 0.0;
    }
    
    static const string& name() {
      static const string _name = "Empty";
      return _name;
    }

};
#endif
