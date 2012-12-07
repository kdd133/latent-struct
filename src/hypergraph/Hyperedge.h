/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _HYPEREDGE_H
#define _HYPEREDGE_H

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "LogWeight.h"
#include "Ublas.h"
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/shared_ptr.hpp>
#include <list>

class Hypernode;
class RealWeight;

class Hyperedge {

public:
  
  Hyperedge(int id, const Hypernode& parent, LogWeight weight,
      SparseLogVec* fv = 0) :
    _id(id), _parent(parent), _weight(weight), _fv(fv) { }
  
  Hyperedge(int id, const Hypernode& parent,
      std::list<const Hypernode*> children, LogWeight weight,
      SparseLogVec* fv = 0) :
    _id(id), _parent(parent), _children(children), _weight(weight), _fv(fv) { }
  
  int getId() const;
  
  const Hypernode& getParent() const;
  
  void setWeight(LogWeight weight);
  
  LogWeight getWeight() const;
  
  const std::list<const Hypernode*>& getChildren() const;
  
  const SparseLogVec* getFeatureVector() const;
  
private:
  int _id;
  const Hypernode& _parent;
  std::list<const Hypernode*> _children;
  LogWeight _weight;
  boost::shared_ptr<SparseLogVec> _fv;
  
};

inline int Hyperedge::getId() const {
  return _id;
}

inline const Hypernode& Hyperedge::getParent() const {
  return _parent;
}

inline void Hyperedge::setWeight(LogWeight weight) {
  _weight = weight;
}

inline LogWeight Hyperedge::getWeight() const {
  return _weight;
}

inline const std::list<const Hypernode*>& Hyperedge::getChildren() const {
  return _children;
}

inline const SparseLogVec* Hyperedge::getFeatureVector() const {
  return _fv.get();
}

#endif
