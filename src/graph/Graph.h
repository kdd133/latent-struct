/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _GRAPH_H
#define _GRAPH_H

#include "FeatureVector.h"
#include "Label.h"
#include <boost/shared_array.hpp>
#include <list>
#include <string>
using namespace boost;
using namespace std;

class FeatureMatrix;
class LogWeight;
class Pattern;
class RealWeight;
class WeightVector;

class Graph {
  public:
    virtual void build(const WeightVector& w, const Pattern& x, Label label,
        bool includeStartArc, bool includeObservedFeaturesArc) = 0;
      
    virtual void rescore(const WeightVector& w) = 0;

    virtual LogWeight logPartition() = 0;

    // Note: Assumes fv has been zeroed out.
    virtual LogWeight logExpectedFeaturesUnnorm(FeatureVector<LogWeight>& fv,
        shared_array<LogWeight> buffer) = 0; 

    // Note: Assumes fv has been zeroed out.
    virtual RealWeight maxFeatureVector(FeatureVector<RealWeight>& fv,
        bool getCostOnly = false) = 0;
        
    virtual LogWeight logExpectedFeatureCooccurrences(FeatureMatrix& fm,
        FeatureVector<LogWeight>& fv) = 0;
        
    // Returns the *reverse* sequence of edit operations in to the maximum
    // scoring alignment. i.e., The operations corresponding to these ids can
    // be applied in reverse order to reconstruct the optimal alignment.
    virtual void maxAlignment(list<int>& opIds) const = 0;
    
    virtual void toGraphviz(const string& fname) const = 0;
    
    virtual int numArcs() = 0;
    
    virtual void clearDynProgVariables() = 0;
    
    virtual void clearBuildVariables() = 0;
};

#endif
