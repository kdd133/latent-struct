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
#include "FeatureGenConstants.h"
#include "FeatureVector.h"
#include "Hyperedge.h"
#include "Hypernode.h"
#include "LogWeight.h"
#include "OpNone.h"
#include "Ring.h"
#include "RingInfo.h"
#include "StateType.h"
#include "StringPair.h"
#include "WeightVector.h"
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <stack>
#include <tr1/unordered_map>
#include <vector>
using namespace boost;
using namespace std;

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
//  _fst = new fst::VectorFst<Arc>();
  
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
    FeatureVector<RealWeight>* fv = _fgenObs->getFeatures(pair, label);
    assert(fv);
    // Note: Even if includeObsArc=true, we omit the observed feature vector in
    // the event that no observed features actually fire.
    if (fv->getLength() > 0) {
      const StateId preStartStateId = addNode();
      addEdge(noOp.getId(), startFinishStateType.getId(), preStartStateId,
          startStateId, fv, w);
      startStateId = preStartStateId;
      _root = &_nodes[_nodes.size()-1];
    }
    else
      delete fv;
  }
//  _fst->SetStart(startStateId);
  
  if (includeStartArc) {
    FeatureVector<RealWeight>* fv = _fgen->getFeatures(pair, label, 0, 0, noOp,
        history);
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
  BOOST_FOREACH(Hyperedge edge, _edges) {
    const double newWeight = w.innerProd(edge.getFeatureVector());
    edge.setWeight(newWeight);
  }
}

void AlignmentHypergraph::getNodesTopologicalOrder(
    list<const Hypernode*>& ordering, bool reverse) {
  assert(ordering.size() == 0);
  ordering.clear();
  
  // True once this node and all its children have been covered
  scoped_array<bool> completed(new bool[_nodes.size()]);
  
  // True after the first time we hit a node
  scoped_array<bool> reached(new bool[_nodes.size()]);
  
  for (size_t i = 0; i < _nodes.size(); i++) {
    completed[i] = false;
    reached[i] = false;
  }
  
  stack<const Hypernode*> stck;
  stck.push(_root);
  
  while (stck.size() > 0)
  {
    const Hypernode* cur = stck.top();
    
    if (completed[cur->getId()]) {
      // Only want to emit each node once
      stck.pop();
    }
    else {
      if (reached[cur->getId()]) {
        // If this is the second time we've hit this node, it's safe to emit
        stck.pop();
        completed[cur->getId()] = true;
        if (reverse)
          ordering.push_back(cur);
        else
          ordering.push_front(cur);
      }
      else {
        // The first time we hit a node, push all children to make sure they're covered
        reached[cur->getId()] = true;
        BOOST_FOREACH(const Hyperedge* edge, cur->getEdges()) {
          BOOST_FOREACH(const Hypernode* child, edge->getChildren()) {
            if (!completed[child->getId()]) {
              assert(!reached[child->getId()]); // Check for cycles
              stck.push(child);
            }
          }
        }
      }
    }
  }
}

shared_array<RingInfo> AlignmentHypergraph::inside(const Ring ring) {
  list<const Hypernode*> revTopOrder;
  getNodesTopologicalOrder(revTopOrder, true);
  assert(revTopOrder.size() == _nodes.size());
  
  shared_array<RingInfo> betas(new RingInfo[_nodes.size()]);
  
  // The beta value for the "root" node (i.e., the goal node in this case, since
  // we are working in reverse) is one by construction.
  betas[_goal->getId()] = RingInfo::one(ring);
  
  // For each node, in reverse topological order...
  BOOST_FOREACH(const Hypernode* v, revTopOrder) {
    if (v == _goal)
      continue; // Skip the goal node, which was handled above.
      
    const int parentId = v->getId();      
    betas[parentId] = RingInfo::zero(ring);
    
    // For each incoming edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      RingInfo k(*e, ring);
      // For each antecedent node...
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        k.collectProd(betas[u->getId()], ring);
      betas[parentId].collectSum(k, ring);
    }
  }
  return betas;
}

shared_array<RingInfo> AlignmentHypergraph::outside(const Ring ring,
    shared_array<RingInfo> betas) {
  list<const Hypernode*> topOrder;
  getNodesTopologicalOrder(topOrder, false);
  assert(topOrder.size() == _nodes.size());
  
  shared_array<RingInfo> alphas(new RingInfo[_nodes.size()]);
  for (size_t i = 0; i < _nodes.size(); ++i)
    alphas[i] = RingInfo::zero(ring);
  
  alphas[_root->getId()] = RingInfo::one(ring);
  
  // For each node, in topological order...
  BOOST_FOREACH(const Hypernode* v, topOrder) {
    // For each outgoing edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      BOOST_FOREACH(const Hypernode* u, e->getChildren()) {
        RingInfo score(*e, ring); // Initialize to the score of the edge
        // Incorporate the product of the sibling beta scores
        BOOST_FOREACH(const Hypernode* w, e->getChildren()) {
          if (w != u)
            score.collectProd(betas[w->getId()], ring);
        }
        score.collectProd(alphas[v->getId()], ring);
        alphas[u->getId()].collectSum(score, ring);
      }
    }
  }
  return alphas;
}

AlignmentHypergraph::InsideOutsideResult AlignmentHypergraph::insideOutside(
    const Ring ring) {

  // Run inside() and outside().
  shared_array<RingInfo> betas = inside(ring);
  shared_array<RingInfo> alphas = outside(ring, betas);
    
  const int d = _fgen->getAlphabet()->size();
  shared_array<LogWeight> array(new LogWeight[d]);
  tr1::unordered_map<int,LogWeight> fvExp;
  shared_ptr<FeatureMatrix> fmExp(new FeatureMatrix(d));
  
  BOOST_FOREACH(const Hypernode& v, _nodes) {
    BOOST_FOREACH(const Hyperedge* e, v.getEdges()) {
      RingInfo keBar(alphas[v.getId()]);      
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        keBar.collectProd(betas[u->getId()], RingLog);

      LogWeight pe = e->getWeight();
      assert(e->getFeatureVector());
      FeatureVector<LogWeight> fv = fvConvert(*e->getFeatureVector(), array, d);
      
      if (ring == RingLog || ring == RingViterbi) {
        // Compute: xHat = xHat + keBar*xe
        //    where xe = pe*re
        FeatureVector<LogWeight>& re = fv; // Interpret fv as re in this case
        pe.timesEquals(keBar.score());
        re.addTo(fvExp, pe);
      }
      else {
        assert(ring == RingExpectation);
        FeatureVector<LogWeight>& se = fv; // Interpret fv as se in this case
        FeatureVector<LogWeight> pese(se);
        pese.timesEquals(pe);
        shared_ptr<FeatureMatrix> pesese = pese.outerProd(se, d);
        
        pese.addTo(fvExp, keBar.score());
        
        pesese->timesEquals(keBar.score()); // pt
        assert(keBar.fv());
        shared_ptr<FeatureMatrix> rs = keBar.fv()->outerProd(pese, d); // rs
        fmExp->logAppend(*pesese);
        fmExp->logAppend(*rs);
      }
    }
  }
  
  InsideOutsideResult result;
  result.Z = betas[_root->getId()].score();
  result.rBar.reset(new FeatureVector<LogWeight>(*betas[_root->getId()].fv()));
  result.sBar.reset(new FeatureVector<LogWeight>(fvExp));
  result.tBar = fmExp;
  
  return result;
}

LogWeight AlignmentHypergraph::logPartition() {
  shared_array<RingInfo> betas = inside(RingLog);
  return betas[_root->getId()].score();
}

LogWeight AlignmentHypergraph::logExpectedFeaturesUnnorm(
    FeatureVector<LogWeight>& fv, shared_array<LogWeight> logArray) {
#ifdef USE_EXP_SEMI // Test the expectation semiring.
  cout << "AlignmentHypergraph: Using RingExpectation in " <<
      "logExpectedFeaturesUnnorm()" << endl;  
  shared_array<RingInfo> betas = inside(RingExpectation);
  tr1::unordered_map<int,LogWeight> temp;
  const FeatureVector<LogWeight>* expFv = betas[_root->getId()].fv();
  for (int i = 0; i < expFv->getNumEntries(); ++i)
    temp[expFv->getIndexAtLocation(i)] = expFv->getValueAtLocation(i);
  fv.reinit(temp);
  return betas[_root->getId()].score();
#else
  // Run inside() and outside().
  shared_array<RingInfo> betas = inside(RingLog);
  shared_array<RingInfo> alphas = outside(RingLog, betas);
  
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
  
  BOOST_FOREACH(const Hypernode& v, _nodes) {
    BOOST_FOREACH(const Hyperedge* e, v.getEdges()) {
      RingInfo ke(alphas[v.getId()]);
      BOOST_FOREACH(const Hypernode* u, e->getChildren()) {
        ke.collectProd(betas[u->getId()], RingLog);
      }
      assert(e->getFeatureVector());
      const FeatureVector<RealWeight>& edgeFv = *e->getFeatureVector();

      // If a zero vector is encountered, there is no need to accumulate it.
      if (edgeFv.getLength() == 0)
        continue;

      FeatureVector<LogWeight> xe = fvConvert(edgeFv, logArray, d);
      LogWeight weight(ke.score());
      weight.timesEquals(e->getWeight());
      xe.addTo(sparse, weight);
    }
  }
  fv.reinit(sparse);

  return betas[_root->getId()].score();
#endif
}

LogWeight AlignmentHypergraph::logExpectedFeatureCooccurrences(
    shared_ptr<FeatureMatrix>& fm, shared_ptr<FeatureVector<LogWeight> >& fv) {
  InsideOutsideResult inOut = insideOutside(RingExpectation);
  fv = inOut.sBar;
  fm = inOut.tBar;
  return inOut.Z;
}

RealWeight AlignmentHypergraph::maxFeatureVector(FeatureVector<RealWeight>& fv,
    bool getCostOnly) {  
#if 0
  // We save and then restore _betas (which will be overwritten by the call to
  // inside()) in case, e.g., logPartition() had already been computed for this
  // graph. That is, we have no reason to store the values for RingViterbi.
  cout << "AlignmentHypergraph: Calling inside() instead of viterbi()" << endl;
  shared_array<RingInfo> betasCopy = _betas;
  inside(RingViterbi);
  // Note: This should yield the same score as the call to viterbi() below.
  const RealWeight viterbiScore = _betas[_root->getId()].score().value();
  _betas = betasCopy;
#else
  list<const Hyperedge*> bestPath;
  const RealWeight viterbiScore = viterbi(bestPath);
  if (!getCostOnly) {
    tr1::unordered_map<int, RealWeight> sparse;
    BOOST_FOREACH(const Hyperedge* e, bestPath) {
      assert(e->getChildren().size() == 1); // Only works for graphs at this point
      const FeatureVector<RealWeight>* edgeFv = e->getFeatureVector();
      assert(edgeFv);
      edgeFv->addTo(sparse);
    }
    fv.reinit(sparse);
  }
#endif
  return viterbiScore;
}

double AlignmentHypergraph::viterbi(list<const Hyperedge*>& path) {
  list<const Hypernode*> revTopOrder;
  getNodesTopologicalOrder(revTopOrder, true);
  assert(revTopOrder.size() == _nodes.size());
  path.clear();
  
  typedef struct entry {
    LogWeight score;
    const Hyperedge* backPointer;
  } Entry;
  
  Entry* chart = new Entry[_nodes.size()];
  for (size_t i = 0; i < _nodes.size(); ++i) {
    chart[i].score = LogWeight::kZero;
    chart[i].backPointer = 0;
  }
  chart[_goal->getId()].score = LogWeight::kOne;
  
  // For each node, in reverse topological order...
  BOOST_FOREACH(const Hypernode* v, revTopOrder) {  
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      LogWeight pathScore = e->getWeight();
      
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        pathScore.timesEquals(chart[u->getId()].score);
      
      Entry& nodeEntry = chart[v->getId()];
      if (!nodeEntry.backPointer || nodeEntry.score < pathScore) {
        nodeEntry.score = pathScore;
        nodeEntry.backPointer = e;
      }
    }
  }
  
  const Hypernode* v = _root;
  while (v != _goal) {
    const Hyperedge* e = chart[v->getId()].backPointer;
    assert(e->getChildren().size() == 1); // Only works for graphs at this point
    path.push_back(e);
    v = e->getChildren().front();
  }
  
  const double pathScore = chart[_root->getId()].score.value();
  delete[] chart;
  return pathScore;
}

void AlignmentHypergraph::maxAlignment(list<int>& opIds) const {
  assert(0);
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

int AlignmentHypergraph::numArcs() {
  return _edges.size();
}

void AlignmentHypergraph::clearDynProgVariables() {

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
    const int numOutgoing = numEdges(sourceStateId);
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
      FeatureVector<RealWeight>* fv = _fgen->getFeatures(pair, label, i, j,
          noOp, history);
      addEdge(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          _goal->getId(), fv, w);
      history.pop_back();
    }
    else {
      addEdge(noOp.getId(), startFinishStateType.getId(), sourceStateId,
          _goal->getId(), new FeatureVector<RealWeight>(), w);
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

      // Append the state and the consumed strings to the alignment history.
      AlignmentPart part = {op->getName(), sourceConsumed, targetConsumed};
      history.push_back(part);
      FeatureVector<RealWeight>* fv = _fgen->getFeatures(pair, label, iNew,
          jNew, *op, history);
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
    const StateId sourceId, const StateId destId, FeatureVector<RealWeight>* fv,
    const WeightVector& w) {
  assert(fv);
  assert(sourceId >= 0);
//  Arc arc(opId, destStateTypeId, (double)-w.innerProd(fv), destId, fv);
  Hypernode& parent = _nodes[sourceId];
  Hypernode* onlyChild = &_nodes[destId]; 
  list<const Hypernode*> children;
  children.push_back(onlyChild);
  const int edgeId = _edges.size();
  const double edgeWeight = w.innerProd(fv);
//  _fst->AddArc(sourceId, arc);
  Hyperedge* edge = new Hyperedge(edgeId, parent, children, edgeWeight, fv);
  parent.addEdge(edge);
  _edges.push_back(edge);
}

void AlignmentHypergraph::clear() {
  _nodes.clear();
  _edges.clear();
  clearDynProgVariables();
}

int AlignmentHypergraph::numEdges(int nodeId) const {
  assert(nodeId < _nodes.size());
  return _nodes[nodeId].getEdges().size();
}
