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

#include "ExpectationSemiring.h"
#include "Graph.h"
#include "Hyperedge.h"
#include "Hypernode.h"
#include "Inference.h"
#include "Label.h"
#include "LogSemiring.h"
#include "LogWeight.h"
#include "Ublas.h"
#include "ViterbiSemiring.h"
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
#include <stack>

template <class Semiring>
class Inference {
  public:
    static LogWeight logPartition(const Graph& g);

    static LogWeight logExpectedFeatures(const Graph& g, LogVec& fv); 
        
    static LogWeight logExpectedFeatureCooccurrences(const Graph& g, SparseLogMat& fm,
        LogVec& fv);
    
    // Note: maxFeatureVector and viterbiScore will return the same value, but
    // if you only need the score, the viterbiScore function should be slightly
    // more efficient because it does not keep track of back-pointers.    
    static double maxFeatureVector(const Graph& g, SparseRealVec& fv);
    static double viterbiScore(const Graph& g);
        
    // Returns the *reverse* sequence of labels that correspond to the edges
    // in the Viterbi (max-scoring) path.
    static void viterbiPath(const Graph& g, std::list<int>& edgeLabels);
    
  private:    
    static void insideOutside(const Graph& g, typename
        Semiring::InsideOutsideResult& result);
    
    static boost::shared_array<Semiring> inside(const Graph& g);
    
    static boost::shared_array<Semiring> outside(const Graph& g,
        boost::shared_array<Semiring> betas);
    
    static double viterbi(const Graph& g, std::list<const Hyperedge*>& bestPath);
    
    static void getNodesTopologicalOrder(const Graph& g,
        std::list<const Hypernode*>& ordering, bool reverse = false);
        
    typedef struct viterbi_entry {
      LogWeight score;
      const Hyperedge* backPointer;
    } ViterbiEntry;
};

template <class Semiring>
LogWeight Inference<Semiring>::logPartition(const Graph& g) {
  boost::shared_array<Semiring> betas = inside(g);
  return betas[g.root()->getId()].score();
}

template <class Semiring>
LogWeight Inference<Semiring>::logExpectedFeatures(const Graph& g, LogVec& fv) {
  typename Semiring::InsideOutsideResult result;
  insideOutside(g, result);
  fv = result.rBar;
  return result.Z;
}

template <class Semiring>
double Inference<Semiring>::maxFeatureVector(const Graph& g,
    SparseRealVec& fv) {
  std::list<const Hyperedge*> bestPath;
  const double viterbiScore = viterbi(g, bestPath);
  
  fv.clear();
  fv.resize(g.numFeatures());
  SparseRealVec temp(fv.size());
  BOOST_FOREACH(const Hyperedge* e, bestPath) {
    assert(e->getChildren().size() == 1); // Only works for graphs at this point
    fv += ublas_util::convertVec(*e->getFeatureVector(), temp);
  }
  return viterbiScore;
}

template <class Semiring>
LogWeight Inference<Semiring>::logExpectedFeatureCooccurrences(const Graph& g,
    SparseLogMat& fm, LogVec& fv) {
  typename Semiring::InsideOutsideResult result;
  insideOutside(g, result);
  fv = result.sBar;
  fm = result.tBar;
  return result.Z;
}

template <class Semiring>
void Inference<Semiring>::viterbiPath(const Graph& g, std::list<int>& labels) {
  labels.clear();
  
  std::list<const Hyperedge*> bestPath;
  viterbi(g, bestPath);
  
  std::list<const Hyperedge*>::const_iterator it;
  for (it = bestPath.begin(); it != bestPath.end(); ++it) {
    assert(*it);
    labels.push_back((*it)->getLabel());
  }
}

template <class Semiring>
double Inference<Semiring>::viterbiScore(const Graph& g) {
  boost::shared_array<Semiring> betas = inside(g);
  return betas[g.root()->getId()].score();
}

template <class Semiring>
void Inference<Semiring>::insideOutside(const Graph& g,
    typename Semiring::InsideOutsideResult& result) {

  // Run inside() and outside().
  boost::shared_array<Semiring> betas = inside(g);
  boost::shared_array<Semiring> alphas = outside(g, betas);
    
  Semiring::initInsideOutsideAccumulator(g.numFeatures(), result);
  
  boost::scoped_array<bool> marked(new bool[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); i++)
    marked[i] = false;

  std::list<const Hypernode*> queue;
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
        
      Semiring keBar(alphas[v->getId()]);      
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        keBar *= betas[u->getId()];

      Semiring::accumulate(result, keBar, *e);
    }
    queue.pop_front(); // Discard the node we just processed.
  }
  
  Semiring::finalizeInsideOutsideResult(result, betas[g.root()->getId()]);
}
    
template <class Semiring>
boost::shared_array<Semiring> Inference<Semiring>::inside(const Graph& g) {
  std::list<const Hypernode*> revTopOrder;
  getNodesTopologicalOrder(g, revTopOrder, true);
  assert(revTopOrder.size() == g.numNodes());
  
  const size_t d = g.numFeatures();
  
  boost::shared_array<Semiring> betas(new Semiring[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); ++i)
    betas[i] = Semiring::zero(d);
  
  // The beta value for the "root" node (i.e., the goal node in this case, since
  // we are working in reverse) is one by construction.
  betas[g.goal()->getId()] = Semiring::one(d);
  
  // For each node, in reverse topological order...
  BOOST_FOREACH(const Hypernode* v, revTopOrder) {
    if (v == g.goal())
      continue; // Skip the goal node, which was handled above.
      
    const int parentId = v->getId();      
    betas[parentId] = Semiring::zero(d);
    
    // For each incoming edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      Semiring k(*e);
      // For each antecedent node...
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        k *= betas[u->getId()];
      betas[parentId] += k;
    }
  }
  return betas;
}

template <class Semiring>
boost::shared_array<Semiring> Inference<Semiring>::outside(const Graph& g,
    boost::shared_array<Semiring> betas) {
  std::list<const Hypernode*> topOrder;
  getNodesTopologicalOrder(g, topOrder, false);
  assert(topOrder.size() == g.numNodes());
  
  const size_t d = g.numFeatures();
  
  boost::shared_array<Semiring> alphas(new Semiring[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); ++i)
    alphas[i] = Semiring::zero(d);  
  alphas[g.root()->getId()] = Semiring::one(d);
  
  // For each node, in topological order...
  BOOST_FOREACH(const Hypernode* v, topOrder) {
    // For each outgoing edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      BOOST_FOREACH(const Hypernode* u, e->getChildren()) {
        Semiring score(*e); // Initialize to the score of the edge
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

template <class Semiring>
double Inference<Semiring>::viterbi(const Graph& g,
    std::list<const Hyperedge*>& path) {
  std::list<const Hypernode*> revTopOrder;
  getNodesTopologicalOrder(g, revTopOrder, true);
  assert(revTopOrder.size() == g.numNodes());
  path.clear();
  
  boost::scoped_array<ViterbiEntry> chart(new ViterbiEntry[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); ++i) {
    chart[i].score = LogWeight();
    chart[i].backPointer = 0;
  }
  chart[g.goal()->getId()].score = LogWeight(1);
  
  // For each node, in reverse topological order...
  BOOST_FOREACH(const Hypernode* v, revTopOrder) {  
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      LogWeight pathScore = e->getWeight();
      
      BOOST_FOREACH(const Hypernode* u, e->getChildren())
        pathScore *= chart[u->getId()].score;
      
      ViterbiEntry& nodeEntry = chart[v->getId()];
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
  return pathScore;
}

template <class Semiring>
void Inference<Semiring>::getNodesTopologicalOrder(const Graph& g,
    std::list<const Hypernode*>& ordering, bool reverse) {
  
  enum colour {BLACK, GREY, WHITE};  
  boost::scoped_array<int> nodeColour(new int[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); i++)
    nodeColour[i] = WHITE;
  
  std::stack<const Hypernode*> theStack;
  const Hypernode* u = g.root();
  nodeColour[u->getId()] = GREY;
  
  ordering.clear();

  while (u != 0) {
    bool uHasWhiteNeighbour = false;
    BOOST_FOREACH(const Hyperedge* edge, u->getEdges()) {
      BOOST_FOREACH(const Hypernode* v, edge->getChildren()) {
        if (nodeColour[v->getId()] == WHITE) {
          uHasWhiteNeighbour = true;
          nodeColour[v->getId()] = GREY;
          theStack.push(u);
          u = v;
          break;
        }
      }
      if (uHasWhiteNeighbour)
        break;
    }
    
    if (!uHasWhiteNeighbour) {
      nodeColour[u->getId()] = BLACK;
      
      if (reverse)
        ordering.push_back(u);
      else
        ordering.push_front(u);
        
      if (theStack.empty())
        u = 0;
      else {
        u = theStack.top();
        theStack.pop();
      }
    }
  }
}

#endif
