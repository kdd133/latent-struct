/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _INFERENCE_H
#define _INFERENCE_H

#include "Label.h"
#include "Ring.h"
#include "Ublas.h"
#include <boost/shared_array.hpp>
#include <list>

class Graph;
class Hyperedge;
class Hypernode;
class LogWeight;
class RingInfo;

class Inference {
  public:
    static LogWeight logPartition(const Graph& g);

    static LogWeight logExpectedFeaturesUnnorm(const Graph& g, LogVec& fv); 

    static double maxFeatureVector(const Graph& g, SparseRealVec& fv,
        bool getCostOnly = false);
        
    static LogWeight logExpectedFeatureCooccurrences(const Graph& g, LogMat& fm,
        LogVec& fv);
        
    // Returns the *reverse* sequence of labels that correspond to the edges
    // in the Viterbi (max-scoring) path.
    static void viterbiPath(const Graph& g, std::list<int>& edgeLabels);
    
  private:
    typedef struct insideOutsideResult {
      LogWeight Z;
      LogVec rBar;
      LogVec sBar;
      LogMat tBar;
    } InsideOutsideResult;
    
    static boost::shared_ptr<InsideOutsideResult> insideOutside(const Graph& g,
        const Ring ring);
    
    static boost::shared_array<RingInfo> inside(const Graph& g, const Ring ring);
    
    static boost::shared_array<RingInfo> outside(const Graph& g,
        const Ring ring, boost::shared_array<RingInfo> betas);
    
    static double viterbi(const Graph& g, std::list<const Hyperedge*>& bestPath);
    
    static void getNodesTopologicalOrder(const Graph& g,
        std::list<const Hypernode*>& ordering, bool reverse = false);
};

#endif
