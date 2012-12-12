/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Graph.h"
#include "Hypernode.h"
#include "Hyperedge.h"
#include "Inference.h"
#include "LogWeight.h"
#include "RingInfo.h"
#include "Ublas.h"
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
#include <stack>

using namespace boost;
using namespace std;

LogWeight Inference::logPartition(const Graph& g) {
  shared_array<RingInfo> betas = inside(g, RingLog);
  return betas[g.root()->getId()].score();
}

LogWeight Inference::logExpectedFeaturesUnnorm(const Graph& g, LogVec& fv) {
  shared_ptr<InsideOutsideResult> inOut = insideOutside(g, RingLog);  
  fv = inOut->rBar;
  return inOut->Z;
}

double Inference::maxFeatureVector(const Graph& g, SparseRealVec& fv,
    bool getCostOnly) {
  list<const Hyperedge*> bestPath;
  const double viterbiScore = viterbi(g, bestPath);
  if (!getCostOnly) {
    fv.clear();
    fv.resize(g.numFeatures());
    SparseRealVec temp(fv.size());
    BOOST_FOREACH(const Hyperedge* e, bestPath) {
      assert(e->getChildren().size() == 1); // Only works for graphs at this point
      fv += ublas_util::convertVec(*e->getFeatureVector(), temp);
    }
  }
  return viterbiScore;
}
        
LogWeight Inference::logExpectedFeatureCooccurrences(const Graph& g, LogMat& fm,
    LogVec& fv) {
  shared_ptr<InsideOutsideResult> inOut = insideOutside(g, RingExpectation);
  fv = inOut->sBar;
  fm = inOut->tBar;
  return inOut->Z;
}

void Inference::viterbiPath(const Graph& g, list<int>& labels) {
  labels.clear();
  
  list<const Hyperedge*> bestPath;
  viterbi(g, bestPath);
  
  list<const Hyperedge*>::const_iterator it;
  for (it = bestPath.begin(); it != bestPath.end(); ++it) {
    assert(*it);
    labels.push_back((*it)->getLabel());
  }
}

shared_ptr<Inference::InsideOutsideResult> Inference::insideOutside(
    const Graph& g, const Ring ring) {
  // Run inside() and outside().
  shared_array<RingInfo> betas = inside(g, ring);
  shared_array<RingInfo> alphas = outside(g, ring, betas);
    
  const int d = g.numFeatures();
  shared_array<LogWeight> array(new LogWeight[d]);
  LogVec rBar(d);
  LogMat tBar(d, d);
  
  scoped_array<bool> marked(new bool[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); i++)
    marked[i] = false;

  list<const Hypernode*> queue;
  queue.push_back(g.root());
  marked[g.root()->getId()] = true;
  
  // Visit the nodes in breadth-first order (order doesn't matter here).
  while (queue.size() > 0) {
    const Hypernode* v = queue.front();
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
    
      // Enqueue this edge's child node(s). 
      BOOST_FOREACH(const Hypernode* child, e->getChildren()) {
        if (!marked[child->getId()]) {
          queue.push_back(child);
          marked[child->getId()] = true;
        }
      }
      
      // If a zero vector is encountered, there is no need to accumulate.
      assert(e->getFeatureVector());
      if (e->getFeatureVector()->size() == 0)
        continue;
        
      RingInfo keBar(alphas[v->getId()]);      
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        keBar *= betas[u->getId()];

      const LogWeight pe = e->getWeight();
      SparseLogVec re = *e->getFeatureVector();
      const LogWeight keBarP = keBar.score();
      
      if (ring == RingLog || ring == RingViterbi) {
        // Compute: xHat = xHat + keBar*xe
        //    where xe = pe*re
        SparseLogVec& pe_re = re;
        pe_re *= pe * keBarP;
        rBar += pe_re;
      }
      else {
        assert(ring == RingExpectation);
        
        SparseLogVec& se = re; // In our applications, we have re == se
        
        SparseLogMat pe_re_se = outer_prod(re, se);
        pe_re_se *= pe;
        
        const SparseLogVec& keBarR = keBar.fv();
        
        SparseLogVec& pe_se = se;
        pe_se *= pe;
        const SparseLogMat pe_se_keBarR = outer_prod(pe_se, keBarR);
        
        pe_se *= keBarP;    // = keBarP * pe_se
        pe_re_se *= keBarP; // = keBarP * pe_re_se
        
        rBar += pe_se;
        tBar += (pe_re_se + pe_se_keBarR);
      }
    }
    queue.pop_front(); // Discard the node we just processed.
  }
  
  shared_ptr<InsideOutsideResult> result(new InsideOutsideResult);
  result->Z = betas[g.root()->getId()].score();
  result->rBar = rBar;
  if (ring == RingExpectation) {
    assert(betas[g.root()->getId()].fv().size() > 0);
    result->sBar = betas[g.root()->getId()].fv();
    result->tBar = tBar;
  }
  
  return result;
}
    
shared_array<RingInfo> Inference::inside(const Graph& g, const Ring ring) {
  list<const Hypernode*> revTopOrder;
  getNodesTopologicalOrder(g, revTopOrder, true);
  assert(revTopOrder.size() == g.numNodes());
  
  const size_t d = g.numFeatures();
  
  shared_array<RingInfo> betas(new RingInfo[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); ++i)
    betas[i] = RingInfo::zero(ring, d);
  
  // The beta value for the "root" node (i.e., the goal node in this case, since
  // we are working in reverse) is one by construction.
  betas[g.goal()->getId()] = RingInfo::one(ring, d);
  
  // For each node, in reverse topological order...
  BOOST_FOREACH(const Hypernode* v, revTopOrder) {
    if (v == g.goal())
      continue; // Skip the goal node, which was handled above.
      
    const int parentId = v->getId();      
    betas[parentId] = RingInfo::zero(ring, d);
    
    // For each incoming edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      RingInfo k(ring, *e);
      // For each antecedent node...
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        k *= betas[u->getId()];
      betas[parentId] += k;
    }
  }
  return betas;
}
    
shared_array<RingInfo> Inference::outside(const Graph& g, const Ring ring,
    boost::shared_array<RingInfo> betas) {
  list<const Hypernode*> topOrder;
  getNodesTopologicalOrder(g, topOrder, false);
  assert(topOrder.size() == g.numNodes());
  
  const size_t d = g.numFeatures();
  
  shared_array<RingInfo> alphas(new RingInfo[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); ++i)
    alphas[i] = RingInfo::zero(ring, d);  
  alphas[g.root()->getId()] = RingInfo::one(ring, d);
  
  // For each node, in topological order...
  BOOST_FOREACH(const Hypernode* v, topOrder) {
    // For each outgoing edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      BOOST_FOREACH(const Hypernode* u, e->getChildren()) {
        RingInfo score(ring, *e); // Initialize to the score of the edge
        // Incorporate the product of the sibling beta scores
        BOOST_FOREACH(const Hypernode* w, e->getChildren()) {
          if (w != u)
            score *= betas[w->getId()];
        }
        score *= alphas[v->getId()];
        alphas[u->getId()] += score;
      }
    }
  }
  return alphas;
}
    
double Inference::viterbi(const Graph& g, std::list<const Hyperedge*>& path) {
  list<const Hypernode*> revTopOrder;
  getNodesTopologicalOrder(g, revTopOrder, true);
  assert(revTopOrder.size() == g.numNodes());
  path.clear();
  
  typedef struct entry {
    LogWeight score;
    const Hyperedge* backPointer;
  } Entry;
  
  Entry* chart = new Entry[g.numNodes()];
  for (size_t i = 0; i < g.numNodes(); ++i) {
    chart[i].score = LogWeight(0);
    chart[i].backPointer = 0;
  }
  chart[g.goal()->getId()].score = LogWeight(1);
  
  // For each node, in reverse topological order...
  BOOST_FOREACH(const Hypernode* v, revTopOrder) {  
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      LogWeight pathScore = e->getWeight();
      
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        pathScore *= chart[u->getId()].score;
      
      Entry& nodeEntry = chart[v->getId()];
      if (!nodeEntry.backPointer || nodeEntry.score < pathScore) {
        nodeEntry.score = pathScore;
        nodeEntry.backPointer = e;
      }
    }
  }
  
  const Hypernode* v = g.root();
  while (v != g.goal()) {
    const Hyperedge* e = chart[v->getId()].backPointer;
    assert(e->getChildren().size() == 1); // Only works for graphs at this point
    path.push_back(e);
    v = e->getChildren().front();
  }
  
  const double pathScore = chart[g.root()->getId()].score;
  delete[] chart;
  return pathScore;
}
    
void Inference::getNodesTopologicalOrder(const Graph& g,
    std::list<const Hypernode*>& ordering, bool reverse) {
  ordering.clear();
  
  // True once this node and all its children have been covered
  scoped_array<bool> completed(new bool[g.numNodes()]);
  
  // True after the first time we hit a node
  scoped_array<bool> reached(new bool[g.numNodes()]);
  
  for (size_t i = 0; i < g.numNodes(); i++) {
    completed[i] = false;
    reached[i] = false;
  }
  
  stack<const Hypernode*> stck;
  stck.push(g.root());
  
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
