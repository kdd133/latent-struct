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
#include <tr1/unordered_map>
#include <vector>
using namespace boost;
using namespace std;


//A transducer that takes two strings as input, and outputs an alignment of the
//two strings. The logSemiring argument indicates whether the Viterbi semiring
//(default) or the Log semiring will be used when building the transducer. This
//in turn determines which inference operations are valid for the transducer;
//i.e., maxFeatureVector() is valid only for the Viterbi semiring, while the
//logPartition() and other "log" operations are only valid for the Log semiring.
template<typename Arc>
class AlignmentTransducer {
  public:
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Weight ArcWeight;
    typedef multi_array<StateId, 3> StateIdTable;
    
    // The first stateType in the list will be used as the start state and as
    // the finish state.
    AlignmentTransducer(const ptr_vector<StateType>& stateTypes,
        shared_ptr<AlignmentFeatureGen> fgen,
        shared_ptr<ObservedFeatureGen> fgenObs,
        bool includeFinalFeats = true);
                        
    ~AlignmentTransducer();
                        
    void build(const WeightVector& w, const StringPair& pair, Label label,
      bool includeObservedFeaturesArc = true);
      
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
    
    const fst::VectorFst<Arc>* getFst() { return _fst; }
    
    int numArcs() { return _numArcs; }
    
    void clearDynProgVariables();
    
    void clearBuildVariables();


  private:
    void applyOperations(const WeightVector& w,
                         const StringPair& pair,
                         const Label label,
                         vector<AlignmentPart>& history,
                         const StateId finishStateId,
                         const int i,
                         const int j);
    
    void addArc(const int opId, const int destStateTypeId,
        const StateId sourceId, const StateId destId,
        FeatureVector<RealWeight>* fv, const WeightVector& w);
        
    void clear();
    
    const ptr_vector<StateType>& _stateTypes;
    
    shared_ptr<AlignmentFeatureGen> _fgen;
    
    shared_ptr<ObservedFeatureGen> _fgenObs;

    fst::VectorFst<Arc>* _fst;
    
    list<const FeatureVector<RealWeight>*> _fvecs;
    
    vector<ArcWeight> _alphas;
    
    vector<ArcWeight> _betas;
    
    StateIdTable _stateIdTable;
    
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
  BOOST_FOREACH(const FeatureVector<RealWeight>* fv, _fvecs) {
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
  // Build lookup tables for state names and edit operation names.
  tr1::unordered_map<int, string> stateNames;
  tr1::unordered_map<int, string> opNames;
  typedef tr1::unordered_map<int, string>::value_type PairType;  
  ptr_vector<StateType>::const_iterator st;
  for (st = _stateTypes.begin(); st != _stateTypes.end(); ++st) {
    stateNames.insert(PairType(st->getId(), st->getName()));
    const ptr_list<EditOperation>& ops = st->getValidOperations();
    ptr_list<EditOperation>::const_iterator op;
    for (op = ops.begin(); op != ops.end(); ++op)
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
        extents[s.size()+1][t.size()+1][_stateTypes.size()]);
  }
  for (size_t i = 0; i <= s.size(); i++)
    for (size_t j = 0; j <= t.size(); j++)
      for (size_t k = 0; k < _stateTypes.size(); k++)
        _stateIdTable[i][j][k] = fst::kNoStateId;
  
  _stateIdTable[0][0][startFinishStateTypeId] = startStateId;
  
  vector<AlignmentPart> history;
  AlignmentPart part = {&startFinishStateType, FeatureGenConstants::EPSILON,
      FeatureGenConstants::EPSILON};
  history.push_back(part);

  // If we have both latent and observed features, we put the observed ones
  // on a "pre-start" arc that every path through the fst must include.
  if (includeObsArc) {
    FeatureVector<RealWeight>* fv = _fgenObs->getFeatures(pair, label);
    const StateId preStartStateId = _fst->AddState();
    addArc(OpNone::ID, startFinishStateTypeId, preStartStateId, startStateId,
        fv, w);
    _fst->SetStart(preStartStateId);
  }
  else
    _fst->SetStart(startStateId);

  applyOperations(w, pair, label, history, finishStateId, 0, 0);
  
  assert(_fvecs.size() > 0);
}

template<typename Arc>
void AlignmentTransducer<Arc>::applyOperations(const WeightVector& w,
    const StringPair& pair, const Label label, vector<AlignmentPart>& history,
    const StateId finishStateId, const int i, const int j) {
  const int S = pair.getSource().size();
  const int T = pair.getTarget().size();
  assert(i <= S && j <= T); // an op should never take us out of bounds
  const StateType& sourceStateType = *history.back().state;
  const StateId& sourceStateId = _stateIdTable[i][j][sourceStateType.getId()];
  
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
    FeatureVector<RealWeight>* fv = 0;
    OpNone noOp;
    if (_includeFinalFeats) {
      // The OpNone doesn't consume any of the strings, hence the epsilons below.
      AlignmentPart part = {&startFinishStateType, FeatureGenConstants::EPSILON,
          FeatureGenConstants::EPSILON};
      history.push_back(part);
      fv = _fgen->getFeatures(pair, label, i, j, noOp, history);
      addArc(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          finishStateId, fv, w);
      history.pop_back();
    }
    else {
      addArc(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          finishStateId, fv, w); // Note: using zero fv
    }
    return;
  }

  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  
  const ptr_list<EditOperation>& ops = sourceStateType.getValidOperations();
  ptr_list<EditOperation>::const_iterator op;
  for (op = ops.begin(); op != ops.end(); ++op) {
    int iNew = -1, jNew = -1;
    const StateType* destStateType = op->apply(s, t, &sourceStateType, i, j,
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
      AlignmentPart part = {destStateType, sourceConsumed, targetConsumed};
      history.push_back(part);      
      FeatureVector<RealWeight>* fv = _fgen->getFeatures(pair, label, iNew,
          jNew, *op, history);
      addArc(op->getId(), destStateType->getId(), sourceStateId, destStateId,
          fv, w);
      applyOperations(w, pair, label, history, finishStateId, iNew, jNew);
      history.pop_back();
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
    FeatureVector<LogWeight>& fv, shared_array<LogWeight> logArray) {
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
  
  // FIXME: We're assuming d doesn't change between calls, but we don't actually
  // verify this. In fact, this whole buffer business is ugly and should be done
  // away with if possible.
  const int d = _fgen->getAlphabet()->size();
  if (!logArray) {
    // The alphabet is shared between the two f-gens (see latent_struct.cpp)
    assert(d == _fgenObs->getAlphabet()->size());
    logArray.reset(new LogWeight[d]);
  }
  assert(logArray);
  
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
      convert(*arc.fv, logArray, d).addTo(sparse, weight);
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
  
  tr1::unordered_map<int, RealWeight> sparse;

  // ShortestPath builds an fst in reverse order, assigning id 0 to the Final
  // state, and then incrementing the id for each state along the path. So, if
  // we start at the Start state, the ids will decrease until we reach 0, at
  // which point we are done.
  assert(viterbiFst.Start() != 0);
  double cost = 0;
  fst::StateIterator<fst::VectorFst<Arc> > sIt(viterbiFst);
  for (; !sIt.Done(); sIt.Next()) {
    fst::ArcIterator<fst::VectorFst<Arc> > aIt(viterbiFst, sIt.Value());
    if (!aIt.Done()) {
      const Arc& arc = aIt.Value();
      // Avoid performing the vector additions if we only want to know the cost.
      if (arc.fv && !getCostOnly)
        arc.fv->addTo(sparse);
      cost += arc.weight.Value();    
      // There should be at most one outgoing arc per state in the Viterbi fst.
      aIt.Next();
      assert(aIt.Done());
    }
  }
  if (!getCostOnly)
    fv.reinit(sparse);
  return RealWeight(-cost); // the arc weights were negated in build()
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
      opIds.push_back(aIt.Value().ilabel);
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
    FeatureVector<LogWeight>& fv, shared_array<LogWeight> buffer) {
  throw logic_error("Can't compute expectations in the Tropical semiring.");
}

template<> inline RealWeight
AlignmentTransducer<LogFeatArc>::maxFeatureVector(
    FeatureVector<RealWeight>& fv, bool getCostOnly) {
  throw logic_error("Can't run Viterbi in the Log semiring.");
}

#endif
