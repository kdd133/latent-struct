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

class LogWeight;
class Pattern;
class RealWeight;
class WeightVector;

class Graph {
  public:
    void build(const WeightVector& w, const Pattern& x, Label label,
        bool includeStartArc, bool includeObservedFeaturesArc);
      
    void rescore(const WeightVector& w);

    LogWeight logPartition();

    // Note: Assumes fv has been zeroed out.
    LogWeight logExpectedFeaturesUnnorm(FeatureVector<LogWeight>& fv,
        shared_array<LogWeight> buffer); 

    // Note: Assumes fv has been zeroed out.
    RealWeight maxFeatureVector(FeatureVector<RealWeight>& fv,
        bool getCostOnly = false);
        
    // Returns the *reverse* sequence of edit operations in to the maximum
    // scoring alignment. i.e., The operations corresponding to these ids can
    // be applied in reverse order to reconstruct the optimal alignment.
    void maxAlignment(list<int>& opIds) const;
    
    void toGraphviz(const string& fname) const;
    
    int numArcs();
    
    void clearDynProgVariables();
    
    void clearBuildVariables();
};

#endif
