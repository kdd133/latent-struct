/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentHypergraph.h"
#include "AlignmentPart.h"
#include "Alphabet.h"
#include "FeatureGenConstants.h"
#include "Hyperedge.h"
#include "Hypernode.h"
#include "LogWeight.h"
#include "OpNone.h"
#include "StateType.h"
#include "StringPair.h"
#include "Ublas.h"
#include "WeightVector.h"
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <fstream>
#include <stack>
#include <vector>

using namespace std;
using namespace boost;

//#define USE_EXP_SEMI // uncomment to use RingExpectation where possible

const AlignmentHypergraph::StateId AlignmentHypergraph::noId = -1;

AlignmentHypergraph::AlignmentHypergraph(const ptr_vector<StateType>& stateTypes,
    shared_ptr<AlignmentFeatureGen> fgen,
    shared_ptr<ObservedFeatureGen> fgenObs,
    bool includeFinalFeats) :
    _stateTypes(stateTypes), _fgen(fgen), _fgenObs(fgenObs),
    _root(0), _goal(0), _includeFinalFeats(includeFinalFeats) {
}

void AlignmentHypergraph::build(const WeightVector& w, const Pattern& x, Label label,
    bool includeStartArc, bool includeObservedFeaturesArc) {
  const StringPair& pair = (StringPair&)x;
  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  
  clear();
  
  StateId startStateId = addNode();
  _root = &_nodes[_nodes.size()-1];
  addNode();
  _goal = &_nodes[_nodes.size()-1];
  
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
        _stateIdTable[i][j][k] = noId;
  
  _stateIdTable[0][0][startFinishStateType.getId()] = startStateId;
  
  OpNone noOp; // Note: The default id and name suffice for our purposes.
  vector<AlignmentPart> history;
  AlignmentPart part = {noOp.getName(), FeatureGenConstants::EPSILON,
      FeatureGenConstants::EPSILON};
  history.push_back(part);

  // If we have both latent and observed features, we put the observed ones
  // on a "pre-start" arc that every path through the fst must include.
  if (includeObservedFeaturesArc) {
    SparseRealVec* fv = _fgenObs->getFeatures(pair, label);
    assert(fv && (fv->size() > 0 || _fgenObs->getAlphabet()->size() == 0));
    const StateId preStartStateId = addNode();
    addEdge(noOp.getId(), startFinishStateType.getId(), preStartStateId,
        startStateId, fv, w);
    startStateId = preStartStateId;
    _root = &_nodes[_nodes.size()-1];
  }
  
  // If there are no edit operations to apply, we don't need to build a graph.
  // Presumably (as the assert below checks), this implies that we have only
  // observed features, and no latent features.
  if (startFinishStateType.getValidOperations().size() == 0) {
    // We must be using only observed features.
    assert(includeObservedFeaturesArc);
    return;
  }

  if (includeStartArc) {
    SparseRealVec* fv = _fgen->getFeatures(pair, label, 0, 0, 0, 0, noOp,
        history);
    assert(fv && (fv->size() > 0 || _fgen->getAlphabet()->size() == 0));
    const StateId preStartStateId = addNode();
    addEdge(noOp.getId(), startFinishStateType.getId(), preStartStateId,
        startStateId, fv, w);
    startStateId = preStartStateId; // Not used below (yet), but just in case.
    _root = &_nodes[_nodes.size()-1];
  }
  
  assert(_root->getId() == startStateId);

  applyOperations(w, pair, label, history, &startFinishStateType, 0, 0);
}
      
void AlignmentHypergraph::rescore(const WeightVector& w) {
  BOOST_FOREACH(Hyperedge& edge, _edges) {
    const LogWeight newWeight(w.innerProd(*edge.getFeatureVector()), true);
    edge.setWeight(newWeight);
  }
}

void AlignmentHypergraph::toGraphviz(const string& fname) const {
  ofstream fout(fname.c_str());
  assert(fout.good());
  
  fout << "digraph G\n{\n";
  BOOST_FOREACH(const Hypernode& node, _nodes) {
    const int prev = node.getId();
    fout << "node" << prev << " [label=" << prev << "];\n";
    BOOST_FOREACH(const Hyperedge* edge, node.getEdges()) {
      BOOST_FOREACH(const Hypernode* child, edge->getChildren()) {
        fout << "node" << prev << " -> node" << child->getId() << " [label=\"("
            << edge->getId() << ") " << edge->getWeight() << "\"];\n";
      }
    }
  }
  fout << "}\n";

  fout.close();
}

int AlignmentHypergraph::numEdges() const {
  return _edges.size();
}

int AlignmentHypergraph::numNodes() const {
  return _nodes.size();
}

const Hypernode* AlignmentHypergraph::root() const {
  assert(_root);
  return _root;
}

int AlignmentHypergraph::numFeatures() const {
  assert(_fgen->getAlphabet() == _fgenObs->getAlphabet());
  return _fgen->getAlphabet()->size();
}

const Hypernode* AlignmentHypergraph::goal() const {
  assert(_goal);
  return _goal;
}

void AlignmentHypergraph::clearBuildVariables() {
  _stateIdTable.resize(extents[0][0][0]);
}

void AlignmentHypergraph::applyOperations(const WeightVector& w,
    const StringPair& pair, const Label label, vector<AlignmentPart>& history,
    const StateType* sourceStateType, const int i, const int j) {
  const int S = pair.getSource().size();
  const int T = pair.getTarget().size();
  assert(i <= S && j <= T); // an op should never take us out of bounds
  const StateId& sourceStateId = _stateIdTable[i][j][sourceStateType->getId()];
  
  if (i == S && j == T) { // reached finish
    // There must be exactly one outgoing arc from any state at position (S,T).
    const int numOutgoing = numOutgoingEdges(sourceStateId);
    if (numOutgoing > 0) {
      assert(numOutgoing == 1);
      return;
    }
    // The type of the first state in the list defines the type of the start
    // state and the finish state.
    const StateType& startFinishStateType = _stateTypes.front();
    assert(startFinishStateType.getName() == "sta");
    OpNone noOp;
    if (_includeFinalFeats) {
      // The OpNone doesn't consume any of the strings, hence the epsilons below.
      AlignmentPart part = {noOp.getName(), FeatureGenConstants::EPSILON,
          FeatureGenConstants::EPSILON};
      history.push_back(part);
      SparseRealVec* fv = _fgen->getFeatures(pair, label, i, j, i, j, noOp,
          history);
      assert(fv && (fv->size() > 0 || _fgen->getAlphabet()->size() == 0));
      addEdge(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          _goal->getId(), fv, w);
      history.pop_back();
    }
    else {
      addEdge(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          _goal->getId(), new SparseRealVec(_fgen->getAlphabet()->size()), w);
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
      assert(iNew >= i && jNew >= j);
      StateId& destStateId = _stateIdTable[iNew][jNew][destStateType->getId()];
      // If the destination state already exists in the fst, we need to check
      // to see if this particular arc is already present, in which case there
      // is no need to continue down this branch (because this is a depth-first
      // search, we know that it has already been explored).
      if (destStateId != noId) {
        bool arcAlreadyPresent = false;
        BOOST_FOREACH(const Hyperedge* edge, _nodes[sourceStateId].getEdges()) {
          if (edge->getChildren().back()->getId() == destStateId) {
//            assert(arc.ilabel == op->getId());
//            assert(arc.olabel == destStateType->getId());
            arcAlreadyPresent = true;
            break;
          }
        }
        if (arcAlreadyPresent)
          continue;
      }
      else {
        destStateId = addNode(); // note: updates the stateIdTable
      }
      
      // Determine the portions of the strings that were consumed by the op. 
      string sourceConsumed = (i == iNew) ? FeatureGenConstants::EPSILON : s[i];
      for (int k = i + 1; k < iNew; k++)
        sourceConsumed += FeatureGenConstants::PHRASE_SEP + s[k];
      string targetConsumed = (j == jNew) ? FeatureGenConstants::EPSILON : t[j];
      for (int k = j + 1; k < jNew; k++)
        targetConsumed += FeatureGenConstants::PHRASE_SEP + t[k];
      assert(sourceConsumed.size() > 0 || targetConsumed.size() > 0);
      
      // By definition, we should never get an EPSILON-EPSILON alignment
      // character; however, it could happen if, e.g., we accidentally use a
      // CognatePairAligner as the reader, since it inserts EPSILON symbols in
      // the strings.
      assert(sourceConsumed != FeatureGenConstants::EPSILON ||
          targetConsumed != FeatureGenConstants::EPSILON);

      // Append the state and the consumed strings to the alignment history.
      AlignmentPart part = {op->getName(), sourceConsumed, targetConsumed};
      history.push_back(part);
      SparseRealVec* fv = _fgen->getFeatures(pair, label, i, j, iNew, jNew, *op,
          history);
      addEdge(op->getId(), destStateType->getId(), sourceStateId, destStateId,
          fv, w);
      applyOperations(w, pair, label, history, destStateType, iNew, jNew);
      history.pop_back();
    }
  }
}

int AlignmentHypergraph::addNode() {
  Hypernode* node = new Hypernode(_nodes.size());
  _nodes.push_back(node);
  return node->getId();
}

void AlignmentHypergraph::addEdge(const int opId, const int destStateTypeId,
    const StateId sourceId, const StateId destId, SparseRealVec* fv,
    const WeightVector& w) {
  assert(fv);
  assert(sourceId >= 0);
  
//  Arc arc(opId, destStateTypeId, (double)-w.innerProd(fv), destId, fv);

  Hypernode& parent = _nodes[sourceId];
  Hypernode* onlyChild = &_nodes[destId]; 
  list<const Hypernode*> children;
  children.push_back(onlyChild);
  
  const int edgeId = _edges.size();
  const LogWeight edgeWeight(w.innerProd(*fv), true);
  SparseLogVec* logFv = new SparseLogVec(fv->size());
  ublas_util::logarithm(*fv, *logFv);
  delete fv;
  
  Hyperedge* edge = new Hyperedge(edgeId, parent, children, edgeWeight, logFv);
  edge->setLabel(opId); // used by Inference::viterbiPath()
  
  parent.addEdge(edge);
  _edges.push_back(edge);
}

void AlignmentHypergraph::clear() {
  _nodes.clear();
  _edges.clear();
}

int AlignmentHypergraph::numOutgoingEdges(int nodeId) const {
  assert(nodeId < _nodes.size());
  return _nodes[nodeId].getEdges().size();
}
