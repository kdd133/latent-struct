/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Graph.h"
#include "Hyperedge.h"
#include "Hypernode.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <list>
#include <stack>


void Graph::getNodesTopologicalOrder(std::list<const Hypernode*>& ordering,
    bool reverse) const {
  
  enum colour {BLACK, GREY, WHITE};  
  boost::scoped_array<int> nodeColour(new int[numNodes()]);
  for (size_t i = 0; i < numNodes(); i++)
    nodeColour[i] = WHITE;
  
  std::stack<const Hypernode*> theStack;
  const Hypernode* u = root();
  nodeColour[u->getId()] = GREY;
  
  ordering.clear();

  while (u != 0) {
    bool uHasWhiteNeighbour = false;
    BOOST_FOREACH(const Hyperedge* edge, u->getEdges()) {
      assert(edge);
      BOOST_FOREACH(const Hypernode* v, edge->getChildren()) {
        assert(v);
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