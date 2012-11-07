/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _HYPERNODE_H
#define _HYPERNODE_H

#include <list>
using namespace std;

class Hyperedge;

class Hypernode {

public:
  Hypernode() : _id(0) { }

  Hypernode(int id) : _id(id) { }
  
  int getId() const;
  
  void addEdge(const Hyperedge* edge);
  
  const list<const Hyperedge*>& getEdges() const;
  
private:
  int _id;
  list<const Hyperedge*> _edges;
};

inline int Hypernode::getId() const {
  return _id;
}

inline void Hypernode::addEdge(const Hyperedge* edge) {
  _edges.push_back(edge);
}

inline const list<const Hyperedge*>& Hypernode::getEdges() const {
  return _edges;
}

#endif
