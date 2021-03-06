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

#include "AlignmentPart.h"
#include "FeatureGen.h"
#include "Label.h"
#include "Ublas.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

class EditOperation;
class Pattern;
class StateType;

class AlignmentFeatureGen : public FeatureGen {

  public:
  
    AlignmentFeatureGen(boost::shared_ptr<Alphabet> a) : FeatureGen(a) {}
    
    virtual ~AlignmentFeatureGen() {}
    
    //i: Current position in the source string.
    //j: Current position in the target string.
    virtual SparseRealVec* getFeatures(const Pattern& x, Label label,
      int iPrev, int jPrev, int i, int j, const EditOperation& op,
      const std::vector<AlignmentPart>& editHistory) = 0;

    virtual double getDefaultFeatureWeight(const std::string& feature,
        Label label) const = 0;
      
    virtual int processOptions(int argc, char** argv) = 0;
};

#endif
