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

#include "Graph.h"
#include "Hyperedge.h"
#include "Hypernode.h"
#include "Inference.h"
#include "Label.h"
#include "LogSemiring.h"
#include "LogWeight.h"
#include "Ublas.h"
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
#include <vector>


template <class Semiring>
class Inference {
  public:
    static LogWeight logPartition(const Graph& g);

    static LogWeight logExpectedFeatures(const Graph& g, SparseLogVec* fv); 
        
    static void logExpectedFeatureCooccurrences(const Graph& g,
        typename Semiring::InsideOutsideResult& result);
        
    static void logExpectedFeatureCooccurrencesSample(const Graph& g,
        int numSamples, typename Semiring::InsideOutsideResult& result);
    
    // Note: maxFeatureVector and viterbiScore will return the same value, but
    // if you only need the score, the viterbiScore function should be slightly
    // more efficient because it does not keep track of back-pointers.    
    static double maxFeatureVector(const Graph& g, SparseRealVec* fv);
    static double viterbiScore(const Graph& g);
        
    static void insideOutside(const Graph& g, typename
        Semiring::InsideOutsideResult& result);
    
    static boost::shared_array<Semiring> inside(const Graph& g);
    
    static boost::shared_array<Semiring> outside(const Graph& g,
        boost::shared_array<Semiring> betas);

    // Returns the sequence of labels that correspond to the edges in the
    // Viterbi (max-scoring) path, as well as the score of the path.
    static double viterbiPath(const Graph& g,
        std::list<const Hyperedge*>& path);

    // Similar to viterbiPath(), but instead returns the k-best paths.
    static void viterbiPathsK(const Graph& g,
        std::vector<std::list<const Hyperedge*> >& paths);
};

template <class Semiring>
LogWeight Inference<Semiring>::logPartition(const Graph& g) {
  boost::shared_array<Semiring> betas = inside(g);
  return betas[g.root()->getId()].score();
}

template <class Semiring>
LogWeight Inference<Semiring>::logExpectedFeatures(const Graph& g,
    SparseLogVec* fv) {
  typename Semiring::InsideOutsideResult result;
  result.rBar = fv;
  insideOutside(g, result);
  return result.Z;
}

template <class Semiring>
double Inference<Semiring>::maxFeatureVector(const Graph& g, SparseRealVec* fv) {
  std::list<const Hyperedge*> bestPath;
  const double score = viterbiPath(g, bestPath);  
  fv->clear();
  BOOST_FOREACH(const Hyperedge* e, bestPath) {
    assert(e->getChildren().size() == 1); // Only works for graphs at this point
    ublas_util::addExponentiated(*e->getFeatureVector(), *fv);
  }  
  return score;
}

template <class Semiring>
void Inference<Semiring>::logExpectedFeatureCooccurrences(const Graph& g,
    typename Semiring::InsideOutsideResult& result) {
  insideOutside(g, result);
}

template <class Semiring>
void Inference<Semiring>::logExpectedFeatureCooccurrencesSample(const Graph& g,
    int numSamples, typename Semiring::InsideOutsideResult& result) {
    
  boost::shared_array<LogSemiring> betas = Inference<LogSemiring>::inside(g);
  
  Semiring::initInsideOutsideAccumulator(g, result);

  const Hypernode* node = g.root();  
  while (node != g.goal()) {
    //LogWeight insideScore = betas[node->getId()].score();
    BOOST_FOREACH(const Hyperedge* e, node->getEdges()) {
      assert(e);
      //LogWeight edgeWeight = e->getWeight();
      BOOST_FOREACH(const Hypernode* child, e->getChildren()) {
        assert(child);
        
      }
    }
  }
}

template <class Semiring>
void Inference<Semiring>::viterbiPathsK(const Graph& g,
    std::vector<std::list<const Hyperedge*> >& paths) {
  boost::shared_array<Semiring> chart = inside(g);
  const Semiring& r = chart[g.root()->getId()];
  assert(r.size() > 0);  
  paths.clear();
  for (int i = 0; i < r.size(); i++) {
    std::list<const Hyperedge*> path; // this will hold the ith-best path
    const Hyperedge* e = r.entries()[i].bp();
    assert(e); // there should be at least one edge
    path.push_back(e);    
    const ViterbiSemiring* from = r.entries()[i].from();
    if (from) {
      e = from->bp();
      // Loop over the "from" back-pointers in the chart entries.
      while (e != 0) {
        path.push_back(e);
        from = from->from();
        e = from->bp();
      }
    }
    paths.push_back(path);
  }
}

template <class Semiring>
double Inference<Semiring>::viterbiPath(const Graph& g,
    std::list<const Hyperedge*>& path) {  
  boost::shared_array<Semiring> chart = inside(g);  
  path.clear();
  const Hypernode* v = g.root();
  assert(v);
  while (v != g.goal()) {
    const Hyperedge* e = chart[v->getId()].bp();
    assert(e);
    assert(e->getChildren().size() == 1); // Only works for graphs at this point
    path.push_back(e);
    v = e->getChildren().front();
  }
  return chart[g.root()->getId()].score();
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
  
  Semiring::initInsideOutsideAccumulator(g, result);
  
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
      assert(e);
    
      // Enqueue this edge's child node(s). 
      BOOST_FOREACH(const Hypernode* child, e->getChildren()) {
        assert(child);
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
      BOOST_FOREACH(const Hypernode* u, e->getChildren()) {
        assert(u);
        keBar *= betas[u->getId()];
      }

      Semiring::accumulate(result, keBar, *e);
    }
    queue.pop_front(); // Discard the node we just processed.
  }
  
  Semiring::finalizeInsideOutsideResult(result, betas[g.root()->getId()]);
}

template <class Semiring>
boost::shared_array<Semiring> Inference<Semiring>::inside(const Graph& g) {
  std::list<const Hypernode*> revTopOrder;
  g.getNodesTopologicalOrder(revTopOrder, true);
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
    assert(v);
    if (v == g.goal())
      continue; // Skip the goal node, which was handled above.
      
    const int parentId = v->getId();      
    betas[parentId] = Semiring::zero(d);
    
    // For each incoming edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      assert(e);
      Semiring k(*e);
      // For each antecedent node...
      BOOST_FOREACH(const Hypernode* u, e->getChildren()) {
        assert(u);
        k *= betas[u->getId()];
      }
      betas[parentId] += k;
    }
  }
  return betas;
}

template <class Semiring>
boost::shared_array<Semiring> Inference<Semiring>::outside(const Graph& g,
    boost::shared_array<Semiring> betas) {
  std::list<const Hypernode*> topOrder;
  g.getNodesTopologicalOrder(topOrder, false);
  assert(topOrder.size() == g.numNodes());
  
  const size_t d = g.numFeatures();
  
  boost::shared_array<Semiring> alphas(new Semiring[g.numNodes()]);
  for (size_t i = 0; i < g.numNodes(); ++i)
    alphas[i] = Semiring::zero(d);  
  alphas[g.root()->getId()] = Semiring::one(d);
  
  // For each node, in topological order...
  BOOST_FOREACH(const Hypernode* v, topOrder) {
    assert(v);
    // For each outgoing edge...
    BOOST_FOREACH(const Hyperedge* e, v->getEdges()) {
      assert(e);
      BOOST_FOREACH(const Hypernode* u, e->getChildren()) {
        Semiring score(*e); // Initialize to the score of the edge
        // Incorporate the product of the sibling beta scores
        BOOST_FOREACH(const Hypernode* w, e->getChildren()) {
          assert(w && u);
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

#endif
