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
#include "Ublas.h"
#include <boost/multi_array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
#include <string>

class LogWeight;
class Pattern;
class StateType;
class StringPair;
class WeightVector;

class AlignmentHypergraph : public Graph {
  public:
    typedef int StateId;
    typedef boost::multi_array<StateId, 3> StateIdTable;
    
    virtual ~AlignmentHypergraph() { }
    
    // The first StateType in the list will be used as the start state and as
    // the finish state.
    AlignmentHypergraph(const boost::ptr_vector<StateType>& stateTypes,
        boost::shared_ptr<AlignmentFeatureGen> fgen,
        boost::shared_ptr<ObservedFeatureGen> fgenObs,
        bool includeFinalFeats = true);
    
    virtual void build(const WeightVector& w, const Pattern& x, Label label,
        bool includeStartArc, bool includeObservedFeaturesArc);
      
    virtual void rescore(const WeightVector& w);

    virtual void toGraphviz(const std::string& fname) const;
    
    virtual int numEdges() const;
    
    virtual int numNodes() const;
    
    virtual const Hypernode* root() const;
    
    virtual const Hypernode* goal() const;
    
    virtual int numFeatures() const;
    
    virtual void clearBuildVariables();
    
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
        
    void clear();
    
    int numOutgoingEdges(int nodeId) const;
    
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
