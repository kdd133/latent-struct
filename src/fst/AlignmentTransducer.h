/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _ALIGNMENTTRANSDUCER_H
#define _ALIGNMENTTRANSDUCER_H

#include "AlignmentFeatureGen.h"
#include "EditOperation.h"
#include "FeatureVector.h"
#include "Label.h"
#include "LogFeatArc.h"
#include "LogWeight.h"
#include "ObservedFeatureGen.h"
#include "OpNone.h"
#include "RealWeight.h"
#include "StateType.h"
#include "StdFeatArc.h"
#include "StringPair.h"
#include "Transducer.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <fst/shortest-distance.h>
#include <fst/shortest-path.h>
#include <fst/vector-fst.h>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <tr1/unordered_map>
#include <vector>
using namespace std;


//A transducer that takes two strings as input, and outputs an alignment of the
//two strings. The logSemiring argument indicates whether the Viterbi semiring
//(default) or the Log semiring will be used when building the transducer. This
//in turn determines which inference operations are valid for the transducer;
//i.e., maxFeatureVector() is valid only for the Viterbi semiring, while the
//logPartition() and other "log" operations are only valid for the Log semiring.
template<typename Arc>
class AlignmentTransducer : public Transducer {
  public:
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Weight ArcWeight;
    typedef boost::multi_array<StateId, 3> StateIdTable;
    
    // The first stateType in the list will be used as the start state and as
    // the finish state.
    AlignmentTransducer(const list<StateType>& stateTypes,
                        const list<const EditOperation*>& ops,
                        boost::shared_ptr<AlignmentFeatureGen> fgen,
                        boost::shared_ptr<ObservedFeatureGen> fgenObs,
                        bool includeFinalFeats = true);
                        
    virtual ~AlignmentTransducer();
                        
    void build(const WeightVector& w, const StringPair& pair, Label label,
      bool includeObservedFeaturesArc = true);
      
    void rescore(const WeightVector& w);

    LogWeight logPartition();

    // Note: Assumes fv has been zeroed out.
    LogWeight logExpectedFeaturesUnnorm(FeatureVector<LogWeight>& fv); 

    // Note: Assumes fv has been zeroed out.
    RealWeight maxFeatureVector(FeatureVector<RealWeight>& fv,
        bool getCostOnly = false);
    
    void toGraphviz(const string& fname);
    
    const fst::VectorFst<Arc>* getFst() { return _fst; }
    
    int numArcs() { return _numArcs; }
    
    void clearDynProgVariables();
    
    void clearBuildVariables();


  private:
    void applyOperations(const WeightVector& w,
                         const StringPair& pair,
                         const int label,
                         vector<int>& stateTypeHistory,
                         const StateId finishStateId,
                         const int i,
                         const int j);
                         
    static const EditOperation& noOp() {
      static const OpNone noop = OpNone();
      return noop;
    }
    
    void addArc(const int opId, const int destStateTypeId,
        const StateId sourceId, const StateId destId,
        FeatureVector<RealWeight>* fv, const WeightVector& w);
        
    void clear();
    
    const list<StateType>& _stateTypes;
    
    const list<const EditOperation*>& _ops;
    
    boost::shared_ptr<AlignmentFeatureGen> _fgen;
    
    boost::shared_ptr<ObservedFeatureGen> _fgenObs;

    fst::VectorFst<Arc>* _fst;
    
    list<const FeatureVector<RealWeight>*> _fvecs;
    
    vector<ArcWeight> _alphas;
    
    vector<ArcWeight> _betas;
    
    StateIdTable _stateIdTable;
    
    int _numArcs;
    
    boost::shared_array<LogWeight> _tempLogWeights;
    
    int _tempLogWeightsLength;
    
    // If true, fire a feature for arcs connecting to the Final state.
    bool _includeFinalFeats;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    AlignmentTransducer(const AlignmentTransducer& x);
    AlignmentTransducer& operator=(const AlignmentTransducer& x);
    
};


template<typename Arc>
AlignmentTransducer<Arc>::AlignmentTransducer(
    const list<StateType>& stateTypes,
    const list<const EditOperation*>& ops,
    boost::shared_ptr<AlignmentFeatureGen> fgen,
    boost::shared_ptr<ObservedFeatureGen> fgenObs,
    bool includeFinalFeats) :
    _stateTypes(stateTypes), _ops(ops), _fgen(fgen), _fgenObs(fgenObs), _fst(0),
    _numArcs(0), _includeFinalFeats(includeFinalFeats) {
}

template<typename Arc>
AlignmentTransducer<Arc>::~AlignmentTransducer() {
  clear();
}

template<typename Arc>
void AlignmentTransducer<Arc>::clear() {    
  BOOST_FOREACH(const FeatureVector<RealWeight>* fv, _fvecs) {
    if (fv != 0) {
      if (!((FeatureVector<RealWeight>*)fv)->release()) // non-const cast
        delete fv; // Only delete fv's that aren't in a memory pool.
    }
  }
  
  if (_fst)
    delete _fst;
    
  clearDynProgVariables();
  _fvecs.clear();
  _numArcs = 0;
}

template<typename Arc>
inline void AlignmentTransducer<Arc>::clearDynProgVariables() {
  _alphas.clear();
  _betas.clear();
}

template<typename Arc>
inline void AlignmentTransducer<Arc>::clearBuildVariables() {
  _stateIdTable.resize(boost::extents[0][0][0]);
}

template<typename Arc>
void AlignmentTransducer<Arc>::toGraphviz(const string& fname) {
  ofstream fout(fname.c_str());
  assert(fout.good());
  
  fout << "digraph G\n{\n";
  fst::StateIterator< fst::VectorFst<Arc> > sIt(*_fst);
  for (; !sIt.Done(); sIt.Next()) {
    const StateId prev = sIt.Value();
    fout << "node" << prev << " [label=" << prev << "];\n";
    fst::ArcIterator< fst::VectorFst<Arc> > aIt(*_fst, prev);
    for (; !aIt.Done(); aIt.Next()) {
      stringstream ss;
      const Arc& arc = aIt.Value();
      if (arc.fv) {
        for (int i = 0; i < arc.fv->getNumEntries(); i++)
          ss << arc.fv->getIndexAtLocation(i) << ",";
      }
      const StateId next = arc.nextstate;
      fout << "node" << prev << " -> node" << next << " [label=\"fv:" <<
          ss.str() << " w:" << arc.weight << "\"];\n";
    }
  }
  fout << "}\n";

  fout.close();
}

template<typename Arc>
void AlignmentTransducer<Arc>::build(const WeightVector& w,
    const StringPair& pair, Label label, bool includeObsArc) {
  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  
  clear();
  _fst = new fst::VectorFst<Arc>();
  
  const StateId startStateId = _fst->AddState();
  const StateId finishStateId = _fst->AddState();
  _fst->SetFinal(finishStateId, 0); // 2nd parameter is the final weight
  
  // The type of the first state in the list defines the type of the start
  // state and the finish state.
  const StateType& startFinishStateType = _stateTypes.front();
  const int startFinishStateTypeId = startFinishStateType.getId();
  
  // See if s.size() or t.size() greater than current table dimensions,
  // and if so, reallocate/resize it. Otherwise, we only need to zero out the
  // entries that are within the size requirements of s and t.
  if (_stateIdTable.shape()[0] < s.size()+1 ||
      _stateIdTable.shape()[1] < t.size()+1) {
    _stateIdTable.resize(
        boost::extents[s.size()+1][t.size()+1][_stateTypes.size()]);
  }
  for (size_t i = 0; i <= s.size(); i++)
    for (size_t j = 0; j <= t.size(); j++)
      for (size_t k = 0; k < _stateTypes.size(); k++)
        _stateIdTable[i][j][k] = fst::kNoStateId;
  
  _stateIdTable[0][0][startFinishStateTypeId] = startStateId;
  
  vector<int> stateTypeHistory;
  stateTypeHistory.push_back(startFinishStateTypeId);

  // If we have both latent and observed features, we put the observed ones
  // on a "pre-start" arc that every path throug the fst must include.
  if (includeObsArc) {
    FeatureVector<RealWeight>* fv = _fgenObs->getFeatures(pair, label);
    const StateId preStartStateId = _fst->AddState();
    addArc(noOp().getId(), startFinishStateTypeId, preStartStateId,
        startStateId, fv, w);
    _fst->SetStart(preStartStateId);
  }
  else
    _fst->SetStart(startStateId);

  applyOperations(w, pair, label, stateTypeHistory, finishStateId, 0, 0);
  
  assert(_fvecs.size() > 0);
}

template<typename Arc>
void AlignmentTransducer<Arc>::applyOperations(const WeightVector& w,
    const StringPair& pair, const int label, vector<int>& stateTypeHistory,
    const StateId finishStateId, const int i, const int j) {
  const int S = pair.getSource().size();
  const int T = pair.getTarget().size();
  assert(i <= S && j <= T); // an op should never take us out of bounds
  
  const int sourceStateTypeId = stateTypeHistory.back();
  const StateId& sourceStateId = _stateIdTable[i][j][sourceStateTypeId];
  
  if (i == S && j == T) { // reached finish
    // There must be exactly one outgoing arc from any state at position (S,T).
    const int numOutgoing = _fst->NumArcs(sourceStateId);
    if (numOutgoing > 0) {
      assert(numOutgoing == 1);
      return;
    }
    // The type of the first state in the list defines the type of the start
    // state and the finish state.
    const int startFinishStateTypeId = _stateTypes.front().getId();
    FeatureVector<RealWeight>* fv = 0;
    if (_includeFinalFeats) {
      stateTypeHistory.push_back(startFinishStateTypeId);
      OpNone noOp;
      fv = _fgen->getFeatures(pair, i, j, i, j, label, noOp, stateTypeHistory);
      addArc(noOp.getId(), startFinishStateTypeId, sourceStateId, finishStateId,
          fv, w);
      stateTypeHistory.pop_back();
    }
    else {
      addArc(noOp().getId(), startFinishStateTypeId, sourceStateId,
          finishStateId, fv, w); // Note: using zero fv
    }
    return;
  }

  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  BOOST_FOREACH(const EditOperation* op, _ops) {
    int iNew = -1, jNew = -1;
    const int destStateTypeId = op->apply(s, t, sourceStateTypeId, i, j, iNew,
        jNew);
    if (destStateTypeId >= 0) { // was the operation successfully applied?
      assert(iNew >= 0 && jNew >= 0);
      StateId& destStateId = _stateIdTable[iNew][jNew][destStateTypeId];
      // If the destination state already exists in the fst, we need to check
      // to see if this particular arc is already present, in which case there
      // is no need to continue down this branch (because this is a depth-first
      // search, we know that it has already been explored).
      if (destStateId != fst::kNoStateId) {
        bool arcAlreadyPresent = false;
        for (fst::ArcIterator< fst::VectorFst<Arc> > it(*_fst, sourceStateId);
            !it.Done(); it.Next()) {
          const Arc& arc = it.Value();
          if (arc.nextstate == destStateId) {
            arcAlreadyPresent = true;
            break;
          }
        }
        if (arcAlreadyPresent)
          continue;
      }
      else {
        destStateId = _fst->AddState(); // note: updates the stateIdTable
      }
      
      stateTypeHistory.push_back(destStateTypeId);
      FeatureVector<RealWeight>* fv =  _fgen->getFeatures(pair, i, j, iNew,
          jNew, label, *op, stateTypeHistory);
      addArc(op->getId(), destStateTypeId, sourceStateId, destStateId, fv, w);
      applyOperations(w, pair, label, stateTypeHistory, finishStateId,
          iNew, jNew);
      stateTypeHistory.pop_back();
    }
  }
}

template<typename Arc>
inline void AlignmentTransducer<Arc>::addArc(const int opId,
    const int destStateTypeId, const StateId sourceId, const StateId destId,
    FeatureVector<RealWeight>* fv, const WeightVector& w) {
  // Note that we negate the innerProd so that the dynamic programming
  // routines (e.g., ShortestPath) will return max instead of min.
  Arc arc(opId, destStateTypeId, (double)-w.innerProd(fv), destId, fv);
  _fst->AddArc(sourceId, arc);
  if (fv)
    _fvecs.push_back(fv);  
  _numArcs++;
}

template<typename Arc>
LogWeight AlignmentTransducer<Arc>::logPartition() {
  assert(_fst);

  if (_betas.size() == 0) {
    fst::ShortestDistance(*_fst, &_betas, true);
    // It can happen that some states that can be reached in the forward
    // direction cannot be reached in the backward direction. If one of these
    // states has an id which is greater than any of the states that are
    // reacahable in the backward direction, it will not have an entry in the
    // _betas vector, meaning that _alphas and _betas will be of different
    // sizes. Padding the _betas vector with zero weights solves this problem.
    int n = _betas.size();
    while (n++ < _fst->NumStates())
      _betas.push_back(ArcWeight::Zero());
  }
  assert(_betas.size() > 0 && (int)_betas.size() == _fst->NumStates());  
  return LogWeight(_betas[_fst->Start()]);
}

template<typename Arc>
LogWeight AlignmentTransducer<Arc>::logExpectedFeaturesUnnorm(
    FeatureVector<LogWeight>& fv) {
  assert(_fst);
  assert(!fv.isDense());
  
  if (_alphas.size() == 0) {
    fst::ShortestDistance(*_fst, &_alphas);
    int n = _alphas.size(); // see long comment above in logPartition()
    while (n++ < _fst->NumStates())
      _alphas.push_back(ArcWeight::Zero());
  }
  const LogWeight logZ = logPartition(); // fills in _betas if necessary
  assert(_alphas.size() > 0 && _betas.size() == _alphas.size());
  
  const int de = _fgen->getMaxEntries() + _fgenObs->getMaxEntries();
  if (!_tempLogWeights) {
    _tempLogWeights.reset(new LogWeight[de]); // passed to convert()
    _tempLogWeightsLength = de;
  }
  assert(de == _tempLogWeightsLength);
  
  tr1::unordered_map<int,LogWeight> sparse;
  
  fst::StateIterator< fst::VectorFst<Arc> > sIt(*_fst);
  for (; !sIt.Done(); sIt.Next()) {
    const StateId prevstate = sIt.Value();
    fst::ArcIterator< fst::VectorFst<Arc> > aIt(*_fst, prevstate);
    for (; !aIt.Done(); aIt.Next()) {
      const Arc& arc = aIt.Value();
      if (!arc.fv)
        continue;
      LogWeight weight(arc.weight);
      weight.timesEquals(_alphas[prevstate]);
      weight.timesEquals(_betas[arc.nextstate]);  
      convert(*arc.fv, _tempLogWeights, de).addTo(sparse, weight);
    }
  }
  fv.reinit(sparse);
  return logZ;
}

template<typename Arc>
RealWeight AlignmentTransducer<Arc>::maxFeatureVector(
    FeatureVector<RealWeight>& fv, bool getCostOnly) {
  assert(_fst);
  assert(getCostOnly || !fv.isDense());
  
  fst::VectorFst<Arc> viterbiFst;
  fst::ShortestPath(*_fst, &viterbiFst);
  
  tr1::unordered_map<int,RealWeight> sparse;

  // ShortestPath builds an fst in reverse order, assigning id 0 to the Final
  // state, and then incrementing the id for each state along the path. So, if
  // we start at the Start state, the ids will decrease until we reach 0, at
  // which point we are done.
  assert(viterbiFst.Start() != 0);
  double cost = 0;
  fst::StateIterator< fst::VectorFst<Arc> > sIt(viterbiFst);
  for (; !sIt.Done(); sIt.Next()) {
    const StateId prevstate = sIt.Value();
    fst::ArcIterator< fst::VectorFst<Arc> > aIt(viterbiFst, prevstate);
    for (; !aIt.Done(); aIt.Next()) {
      const Arc& arc = aIt.Value();
      // Avoid performing the vector additions if we only want to know the cost.
      if (arc.fv && !getCostOnly)
        arc.fv->addTo(sparse);
      cost += arc.weight.Value();
    }
  }
  if (!getCostOnly)
    fv.reinit(sparse);
  return RealWeight(-cost); // the arc weights were negated in build()
}

template<typename Arc>
void AlignmentTransducer<Arc>::rescore(const WeightVector& w) {
  fst::StateIterator<fst::VectorFst<Arc> > sit(*_fst);
  for (; !sit.Done(); sit.Next()) {
    StateId state = sit.Value();
    fst::MutableArcIterator<fst::VectorFst<Arc> > ait(_fst, state);
    for (; !ait.Done(); ait.Next()) {
      Arc arcCopy = ait.Value();
      if (!arcCopy.fv)
        continue; // no need to rescore a zero vector
      arcCopy.weight = (double)-w.innerProd(arcCopy.fv); // Note: negating score
      ait.SetValue(arcCopy);
    }
  }
  clearDynProgVariables(); // the cached values are now stale/invalid
}

/* The specializations below make explicit the fact that certain combinations
 * of Arcs and Weights are illegal.
 */

template<> inline LogWeight
AlignmentTransducer<StdFeatArc>::logPartition() {
  throw logic_error(
      "Can't compute the partition function in the Tropical semiring.");
}

template<> inline LogWeight
AlignmentTransducer<StdFeatArc>::logExpectedFeaturesUnnorm(
    FeatureVector<LogWeight>& fv) {
  throw logic_error("Can't compute expectations in the Tropical semiring.");
}

template<> inline RealWeight
AlignmentTransducer<LogFeatArc>::maxFeatureVector(
    FeatureVector<RealWeight>& fv, bool getCostOnly) {
  throw logic_error("Can't run Viterbi in the Log semiring.");
}

#endif
