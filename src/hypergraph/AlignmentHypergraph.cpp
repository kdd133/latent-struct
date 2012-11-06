/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentHypergraph.h"

AlignmentHypergraph::AlignmentHypergraph(const ptr_vector<StateType>& stateTypes,
    shared_ptr<AlignmentFeatureGen> fgen,
    shared_ptr<ObservedFeatureGen> fgenObs,
    bool includeFinalFeats) {
    
}

AlignmentHypergraph::~AlignmentHypergraph() {

}

void AlignmentHypergraph::build(const WeightVector& w, const Pattern& x, Label label,
    bool includeStartArc, bool includeObservedFeaturesArc) {
    
}
      
void AlignmentHypergraph::rescore(const WeightVector& w) {

}

LogWeight AlignmentHypergraph::logPartition() {

}

LogWeight AlignmentHypergraph::logExpectedFeaturesUnnorm(FeatureVector<LogWeight>& fv,
    shared_array<LogWeight> buffer) {
    
}

RealWeight AlignmentHypergraph::maxFeatureVector(FeatureVector<RealWeight>& fv,
    bool getCostOnly) {
    
}
        
void AlignmentHypergraph::maxAlignment(list<int>& opIds) const {

}

void AlignmentHypergraph::toGraphviz(const string& fname) const {

}

int AlignmentHypergraph::numArcs() {

}

void AlignmentHypergraph::clearDynProgVariables() {

}

void AlignmentHypergraph::clearBuildVariables() {

}
