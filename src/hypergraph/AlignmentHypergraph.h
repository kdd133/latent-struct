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

#include "AlignmentFeatureGen.h"
#include "AlignmentPart.h"
#include "Graph.h"
#include "Hyperedge.h"
#include "Hypernode.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include "Ring.h"
#include "Ublas.h"
#include <boost/multi_array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
#include <string>

class LogWeight;
class Pattern;
class RingInfo;
class StateType;
class StringPair;
class WeightVector;

class AlignmentHypergraph : public Graph {
  public:
    typedef int StateId;
    typedef boost::multi_array<StateId, 3> StateIdTable;
    
    typedef struct insideOutsideResult {
      LogWeight Z;
      LogVec rBar;
      LogVec sBar;
      LogMat tBar;
    } InsideOutsideResult;
    
    virtual ~AlignmentHypergraph() { }
    
    // The first StateType in the list will be used as the start state and as
    // the finish state.
    AlignmentHypergraph(const boost::ptr_vector<StateType>& stateTypes,
        boost::shared_ptr<AlignmentFeatureGen> fgen,
        boost::shared_ptr<ObservedFeatureGen> fgenObs,
        bool includeFinalFeats = true);
    
    void build(const WeightVector& w, const Pattern& x, Label label,
        bool includeStartArc, bool includeObservedFeaturesArc);
      
    void rescore(const WeightVector& w);

    void getNodesTopologicalOrder(std::list<const Hypernode*>& ordering,
      bool reverse = false) const;

    LogWeight logPartition();

    // Note: Assumes fv has been zeroed out.
    LogWeight logExpectedFeaturesUnnorm(LogVec& fv);
        
    LogWeight logExpectedFeatureCooccurrences(LogMat& fm, LogVec& fv);

    // Note: Assumes fv has been zeroed out.
    double maxFeatureVector(SparseRealVec& fv, bool getCostOnly = false) const;
        
    // Returns the *reverse* sequence of edit operations in to the maximum
    // scoring alignment. i.e., The operations corresponding to these ids can
    // be applied in reverse order to reconstruct the optimal alignment.
    void maxAlignment(std::list<int>& opIds) const;
    
    void toGraphviz(const std::string& fname) const;
    
    int numArcs();
    
    void clearDynProgVariables();
    
    void clearBuildVariables();
    
    static const StateId noId;
    
  private:
    void applyOperations(const WeightVector& w,
                         const StringPair& pair,
                         const Label label,
                         std::vector<AlignmentPart>& history,
                         const StateType* sourceStateType,
                         const int i,
                         const int j);
    
    int addNode();
    
    void addEdge(const int opId, const int destStateTypeId,
        const StateId sourceId, const StateId destId,
        SparseRealVec* fv, const WeightVector& w);
        
    boost::shared_array<RingInfo> inside(const Ring ring) const;
    
    boost::shared_array<RingInfo> outside(const Ring ring,
        boost::shared_array<RingInfo> betas) const;
        
    InsideOutsideResult insideOutside(const Ring ring) const;
    
    double viterbi(std::list<const Hyperedge*>& bestPath) const;
        
    void clear();
    
    int numEdges(int nodeId) const;
    
    boost::ptr_vector<Hypernode> _nodes;
    
    boost::ptr_vector<Hyperedge> _edges;
    
    const boost::ptr_vector<StateType>& _stateTypes;
    
    boost::shared_ptr<AlignmentFeatureGen> _fgen;
    
    boost::shared_ptr<ObservedFeatureGen> _fgenObs;

    StateIdTable _stateIdTable;
    
    Hypernode* _root;
    
    Hypernode* _goal;
    
    // If true, fire a feature for arcs connecting to the Final state.
    bool _includeFinalFeats;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    AlignmentHypergraph(const AlignmentHypergraph& x);
    AlignmentHypergraph& operator=(const AlignmentHypergraph& x);
};

#endif
