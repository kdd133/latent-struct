/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Hypergraph.h"

void Hypergraph::build(const WeightVector& w, const Pattern& x, Label label,
    bool includeStartArc, bool includeObservedFeaturesArc) {
    
}
      
void Hypergraph::rescore(const WeightVector& w) {

}

LogWeight Hypergraph::logPartition() {

}

LogWeight Hypergraph::logExpectedFeaturesUnnorm(FeatureVector<LogWeight>& fv,
    shared_array<LogWeight> buffer) {
    
}

RealWeight Hypergraph::maxFeatureVector(FeatureVector<RealWeight>& fv,
    bool getCostOnly) {
    
}
        
void Hypergraph::maxAlignment(list<int>& opIds) const {

}

void Hypergraph::toGraphviz(const string& fname) const {

}

int Hypergraph::numArcs() {

}

void Hypergraph::clearDynProgVariables() {

}

void Hypergraph::clearBuildVariables() {

}
