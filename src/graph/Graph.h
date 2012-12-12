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

#include "Label.h"
#include "Ublas.h"
#include <list>
#include <string>

class Hypernode;
class LogWeight;
class Pattern;
class WeightVector;

class Graph {
  public:
    virtual ~Graph() { }
    
    virtual void build(const WeightVector& w, const Pattern& x, Label label,
        bool includeStartArc, bool includeObservedFeaturesArc) = 0;
      
    virtual void rescore(const WeightVector& w) = 0;

    virtual void toGraphviz(const std::string& fname) const = 0;
    
    virtual int numEdges() const = 0;
    
    virtual int numNodes() const = 0;
    
    virtual const Hypernode* root() const = 0;
    
    virtual const Hypernode* goal() const = 0;
    
    virtual int numFeatures() const = 0;
    
    virtual void clearBuildVariables() = 0;
};

#endif
