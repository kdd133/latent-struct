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
#include "AlignmentPart.h"
#include "EditOperation.h"
#include "FeatureGenConstants.h"
#include "Graph.h"
#include "Label.h"
#include "LogFeatArc.h"
#include "LogWeight.h"
#include "ObservedFeatureGen.h"
#include "OpNone.h"
#include "StateType.h"
#include "StdFeatArc.h"
#include "StringPair.h"
#include "Ublas.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
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
#include <string>
#include <vector>

using namespace boost;
using namespace std;

//A transducer that takes two strings as input, and outputs an alignment of the
//two strings.
template<typename Arc>
class AlignmentTransducer : public Graph {
  public:
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Weight ArcWeight;
    typedef multi_array<StateId, 3> StateIdTable;
    
    // The first StateType in the list will be used as the start state and as
    // the finish state.
    AlignmentTransducer(const ptr_vector<StateType>& stateTypes,
        shared_ptr<AlignmentFeatureGen> fgen,
        shared_ptr<ObservedFeatureGen> fgenObs,
        bool includeFinalFeats = true);
                        
    virtual ~AlignmentTransducer();
                        
    virtual void build(const WeightVector& w, const Pattern& x, Label label,
      bool includeStartArc, bool includeObservedFeaturesArc);
      
    virtual void rescore(const WeightVector& w);

    virtual LogWeight logPartition();

    // Note: Assumes fv has been zeroed out.
    virtual LogWeight logExpectedFeaturesUnnorm(LogVec& fv);
        
    virtual LogWeight logExpectedFeatureCooccurrences(LogMat& fm, LogVec& fv);

    // Note: Assumes fv has been zeroed out.
    virtual double maxFeatureVector(SparseRealVec& fv,
        bool getCostOnly = false);
        
    // Returns the sequence of edit operations that constitute the maximum
    // scoring alignment. i.e., The operations corresponding to these ids can
    // be applied in sequential order to reconstruct the optimal alignment.
    virtual void maxAlignment(list<int>& opIds) const;
    
    virtual void toGraphviz(const string& fname) const;
    
    virtual int numArcs() { return _numArcs; }
    
    virtual void clearDynProgVariables();
    
    virtual void clearBuildVariables();


  private:
    void applyOperations(const WeightVector& w,
                         const StringPair& pair,
                         const Label label,
                         vector<AlignmentPart>& history,
                         const StateType* sourceStateType,
                         const int i,
                         const int j);
    
    void addArc(const int opId, const int destStateTypeId,
        const StateId sourceId, const StateId destId,
        SparseRealVec* fv, const WeightVector& w);
        
    void clear();
    
    const ptr_vector<StateType>& _stateTypes;
    
    shared_ptr<AlignmentFeatureGen> _fgen;
    
    shared_ptr<ObservedFeatureGen> _fgenObs;

    fst::VectorFst<Arc>* _fst;
    
    list<const SparseRealVec*> _fvecs;
    
    vector<ArcWeight> _alphas;
    
    vector<ArcWeight> _betas;
    
    StateIdTable _stateIdTable;
    
    StateId _finishStateId;
    
    int _numArcs;
    
    // If true, fire a feature for arcs connecting to the Final state.
    bool _includeFinalFeats;
    
    // private copy constructor and assignment operator (passing by value is
    // not supported for this class)
    AlignmentTransducer(const AlignmentTransducer& x);
    AlignmentTransducer& operator=(const AlignmentTransducer& x);
    
};


template<typename Arc>
AlignmentTransducer<Arc>::AlignmentTransducer(
    const ptr_vector<StateType>& stateTypes,
    shared_ptr<AlignmentFeatureGen> fgen,
    shared_ptr<ObservedFeatureGen> fgenObs,
    bool includeFinalFeats) :
    _stateTypes(stateTypes), _fgen(fgen), _fgenObs(fgenObs), _fst(0),
    _numArcs(0), _includeFinalFeats(includeFinalFeats) {
}

template<typename Arc>
AlignmentTransducer<Arc>::~AlignmentTransducer() {
  clear();
}

template<typename Arc>
void AlignmentTransducer<Arc>::clear() {    
  BOOST_FOREACH(const SparseRealVec* fv, _fvecs) {
    if (fv != 0)
      delete fv;
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
  _stateIdTable.resize(extents[0][0][0]);
}

template<typename Arc>
void AlignmentTransducer<Arc>::toGraphviz(const string& fname) const {
  using namespace std;
  // Build lookup tables for state names and edit operation names.
  tr1::unordered_map<int, string> stateNames;
  tr1::unordered_map<int, string> opNames;
  typedef tr1::unordered_map<int, string>::value_type PairType;  
  ptr_vector<StateType>::const_iterator st;
  for (st = _stateTypes.begin(); st != _stateTypes.end(); ++st) {
    stateNames.insert(PairType(st->getId(), st->getName()));
    BOOST_FOREACH(const EditOperation* op, st->getValidOperations())
      opNames.insert(PairType(op->getId(), op->getName()));
  }

  ofstream fout(fname.c_str());
  assert(fout.good());
  
  fout << "digraph G\n{\n";
  fst::StateIterator< fst::VectorFst<Arc> > sIt(*_fst);
  for (; !sIt.Done(); sIt.Next()) {
    const StateId prev = sIt.Value();
    fout << "node" << prev << " [label=" << prev << "];\n";
    fst::ArcIterator< fst::VectorFst<Arc> > aIt(*_fst, prev);
    for (; !aIt.Done(); aIt.Next()) {
      const Arc& arc = aIt.Value();
      fout << "node" << prev << " -> node" << arc.nextstate << " [label=\"op:"
        << opNames[arc.ilabel] << " st:" << stateNames[arc.olabel] << "\"];\n";
    }
  }
  fout << "}\n";

  fout.close();
}

template<typename Arc>
void AlignmentTransducer<Arc>::build(const WeightVector& w,
    const Pattern& x, Label label, bool startArc, bool obsArc) {
  const StringPair& pair = (StringPair&)x;
  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  
  clear();
  _fst = new fst::VectorFst<Arc>();
  
  StateId startStateId = _fst->AddState();
  _finishStateId = _fst->AddState();
  _fst->SetFinal(_finishStateId, 0); // 2nd parameter is the final weight
  
  // The type of the first state in the list defines the type of the start
  // state and the finish state.
  const StateType& startFinishStateType = _stateTypes.front();
  
  // See if s.size() or t.size() greater than current table dimensions,
  // and if so, reallocate/resize it. Otherwise, we only need to zero out the
  // entries that are within the size requirements of s and t.
  if (_stateIdTable.shape()[0] < s.size()+1 ||
      _stateIdTable.shape()[1] < t.size()+1) {
    _stateIdTable.resize(
        extents[s.size()+1][t.size()+1][_stateTypes.size()]);
  }
  for (size_t i = 0; i <= s.size(); i++)
    for (size_t j = 0; j <= t.size(); j++)
      for (size_t k = 0; k < _stateTypes.size(); k++)
        _stateIdTable[i][j][k] = fst::kNoStateId;
  
  _stateIdTable[0][0][startFinishStateType.getId()] = startStateId;
  
  OpNone noOp; // Note: The default id and name suffice for our purposes.
  vector<AlignmentPart> history;
  AlignmentPart part = {noOp.getName(), FeatureGenConstants::EPSILON,
      FeatureGenConstants::EPSILON};
  history.push_back(part);

  // If we have both latent and observed features, we put the observed ones
  // on a "pre-start" arc that every path through the fst must include.
  if (obsArc) {
    SparseRealVec* fv = _fgenObs->getFeatures(pair, label);
    assert(fv);
    const StateId preStartStateId = _fst->AddState();
    addArc(noOp.getId(), startFinishStateType.getId(), preStartStateId,
        startStateId, fv, w);
    startStateId = preStartStateId;
  }
  _fst->SetStart(startStateId);
  
  if (startArc) {
    SparseRealVec* fv = _fgen->getFeatures(pair, label, 0, 0, noOp, history);
    const StateId preStartStateId = _fst->AddState();
    addArc(noOp.getId(), startFinishStateType.getId(), preStartStateId,
        startStateId, fv, w);
    startStateId = preStartStateId; // Not used below (yet), but just in case.
  }

  applyOperations(w, pair, label, history, &startFinishStateType, 0, 0);
  
  assert(_fvecs.size() > 0);
}

template<typename Arc>
void AlignmentTransducer<Arc>::applyOperations(const WeightVector& w,
    const StringPair& pair, const Label label, vector<AlignmentPart>& history,
    const StateType* sourceStateType, const int i, const int j) {
  const int S = pair.getSource().size();
  const int T = pair.getTarget().size();
  assert(i <= S && j <= T); // an op should never take us out of bounds
  const StateId& sourceStateId = _stateIdTable[i][j][sourceStateType->getId()];
  
  if (i == S && j == T) { // reached finish
    // There must be exactly one outgoing arc from any state at position (S,T).
    const int numOutgoing = _fst->NumArcs(sourceStateId);
    if (numOutgoing > 0) {
      assert(numOutgoing == 1);
      return;
    }
    // The type of the first state in the list defines the type of the start
    // state and the finish state.
    const StateType& startFinishStateType = _stateTypes.front();
    assert(startFinishStateType.getName() == "sta");
    SparseRealVec* fv = 0;
    OpNone noOp;
    if (_includeFinalFeats) {
      // The OpNone doesn't consume any of the strings, hence the epsilons below.
      AlignmentPart part = {noOp.getName(), FeatureGenConstants::EPSILON,
          FeatureGenConstants::EPSILON};
      history.push_back(part);
      fv = _fgen->getFeatures(pair, label, i, j, noOp, history);
      addArc(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          _finishStateId, fv, w);
      history.pop_back();
    }
    else {
      addArc(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          _finishStateId, fv, w); // Note: using zero fv
    }
    return;
  }

  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  
  BOOST_FOREACH(const EditOperation* op, sourceStateType->getValidOperations()) {
    int iNew = -1, jNew = -1;
    const StateType* destStateType = op->apply(s, t, sourceStateType, i, j,
        iNew, jNew);
    if (destStateType != 0) { // was the operation successfully applied?
      assert(iNew >= 0 && jNew >= 0);
      StateId& destStateId = _stateIdTable[iNew][jNew][destStateType->getId()];
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
            assert(arc.ilabel == op->getId());
            assert(arc.olabel == destStateType->getId());
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
      
      // Determine the portions of the strings that were consumed by the op. 
      string sourceConsumed = (i == iNew) ? FeatureGenConstants::EPSILON : s[i];
      for (int k = i + 1; k < iNew; k++)
        sourceConsumed += FeatureGenConstants::PHRASE_SEP + s[k];
      string targetConsumed = (j == jNew) ? FeatureGenConstants::EPSILON : t[j];
      for (int k = j + 1; k < jNew; k++)
        targetConsumed += FeatureGenConstants::PHRASE_SEP + t[k];
      assert(sourceConsumed.size() > 0 || targetConsumed.size() > 0);

      // Append the state and the consumed strings to the alignment history.
      AlignmentPart part = {op->getName(), sourceConsumed, targetConsumed};
      history.push_back(part);
      SparseRealVec* fv = _fgen->getFeatures(pair, label, iNew, jNew, *op,
          history);
      addArc(op->getId(), destStateType->getId(), sourceStateId, destStateId,
          fv, w);
      applyOperations(w, pair, label, history, destStateType, iNew, jNew);
      history.pop_back();
    }
  }
}

template<typename Arc>
inline void AlignmentTransducer<Arc>::addArc(const int opId,
    const int destStateTypeId, const StateId sourceId, const StateId destId,
    SparseRealVec* fv, const WeightVector& w) {
  // Note that we negate the innerProd so that the dynamic programming
  // routines (e.g., ShortestPath) will return max instead of min.
  Arc arc(opId, destStateTypeId, (double)-w.innerProd(*fv), destId, fv);
  assert(sourceId >= 0);
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
  return LogWeight(-_betas[_fst->Start()].Value(), true);
}

template<typename Arc>
LogWeight AlignmentTransducer<Arc>::logExpectedFeaturesUnnorm(LogVec& fv) {
  assert(_fst);
  assert(fv.size() == _fgen->getAlphabet()->size());
  
  if (_alphas.size() == 0) {
    fst::ShortestDistance(*_fst, &_alphas);
    int n = _alphas.size(); // see long comment above in logPartition()
    while (n++ < _fst->NumStates())
      _alphas.push_back(ArcWeight::Zero());
  }
  const LogWeight logZ = logPartition(); // fills in _betas if necessary
  assert(_alphas.size() > 0 && _betas.size() == _alphas.size());
  
  SparseLogVec temp(fv.size());
  
  fv.clear();
  fst::StateIterator< fst::VectorFst<Arc> > sIt(*_fst);
  for (; !sIt.Done(); sIt.Next()) {
    const StateId prevstate = sIt.Value();
    fst::ArcIterator< fst::VectorFst<Arc> > aIt(*_fst, prevstate);
    for (; !aIt.Done(); aIt.Next()) {
      const Arc& arc = aIt.Value();
      if (!arc.fv)
        continue;
      LogWeight weight(-arc.weight.Value(), true);
      weight *= LogWeight(-_alphas[prevstate].Value(), true);
      weight *= LogWeight(-_betas[arc.nextstate].Value(), true);
      fv += (ublas_util::convertVec(*arc.fv, temp) * weight);
    }
  }
  return logZ;
}

template<typename Arc>
LogWeight AlignmentTransducer<Arc>::logExpectedFeatureCooccurrences(LogMat& fm,
    LogVec& fv) {
  assert(0); // Not implemented.
}

template<typename Arc>
double AlignmentTransducer<Arc>::maxFeatureVector(SparseRealVec& fv,
    bool getCostOnly) {
  assert(_fst);
  
  fst::VectorFst<Arc> viterbiFst;
  fst::ShortestPath(*_fst, &viterbiFst);
  
  // ShortestPath builds an fst in reverse order, assigning id 0 to the Final
  // state, and then incrementing the id for each state along the path. So, if
  // we start at the Start state, the ids will decrease until we reach 0, at
  // which point we are done.
  fv.clear();
  assert(viterbiFst.Start() != 0);
  double cost = 0;
  fst::StateIterator<fst::VectorFst<Arc> > sIt(viterbiFst);
  for (; !sIt.Done(); sIt.Next()) {
    fst::ArcIterator<fst::VectorFst<Arc> > aIt(viterbiFst, sIt.Value());
    if (!aIt.Done()) {
      const Arc& arc = aIt.Value();
      // Avoid performing the vector additions if we only want to know the cost.
      if (arc.fv && !getCostOnly)
        fv += *arc.fv;
      cost += arc.weight.Value();    
      // There should be at most one outgoing arc per state in the Viterbi fst.
      aIt.Next();
      assert(aIt.Done());
    }
  }
  return -cost; // the arc weights were negated in build()
}

template<typename Arc>
void AlignmentTransducer<Arc>::maxAlignment(list<int>& opIds) const {
  assert(_fst);
  opIds.clear();
  
  fst::VectorFst<Arc> viterbiFst;
  fst::ShortestPath(*_fst, &viterbiFst);
  assert(viterbiFst.Start() != 0);
  
  fst::StateIterator<fst::VectorFst<Arc> > sIt(viterbiFst);
  for (; !sIt.Done(); sIt.Next()) {
    const StateId stateId = sIt.Value();
    fst::ArcIterator<fst::VectorFst<Arc> > aIt(viterbiFst, stateId);
    if (!aIt.Done()) {
      opIds.push_front(aIt.Value().ilabel);
      // There should be at most one outgoing arc per state in the Viterbi fst.
      aIt.Next();
      assert(aIt.Done());
    }
  }
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
      arcCopy.weight = (double)-w.innerProd(*arcCopy.fv); // Note: negating score
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
AlignmentTransducer<StdFeatArc>::logExpectedFeaturesUnnorm(LogVec& fv) {
  throw logic_error("Can't compute expectations in the Tropical semiring.");
}

template<> inline double
AlignmentTransducer<LogFeatArc>::maxFeatureVector(SparseRealVec& fv,
    bool getCostOnly) {
  throw logic_error("Can't run Viterbi in the Log semiring.");
}

#endif
