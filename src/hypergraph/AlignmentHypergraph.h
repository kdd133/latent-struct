/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _ALIGNMENTHYPERGRAPH_H
#define _ALIGNMENTHYPERGRAPH_H

#include "FeatureVector.h"
#include "Graph.h"
#include "Label.h"
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
#include <string>
using namespace boost;
using namespace std;

class AlignmentFeatureGen;
class LogWeight;
class ObservedFeatureGen;
class Pattern;
class RealWeight;
class StateType;
class WeightVector;

class AlignmentHypergraph : public Graph {
  public:
    // The first StateType in the list will be used as the start state and as
    // the finish state.
    AlignmentHypergraph(const ptr_vector<StateType>& stateTypes,
        shared_ptr<AlignmentFeatureGen> fgen,
        shared_ptr<ObservedFeatureGen> fgenObs,
        bool includeFinalFeats = true);
                        
    ~AlignmentHypergraph();
    
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
