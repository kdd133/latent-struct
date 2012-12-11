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

class Hyperedge {

public:
  
  Hyperedge(int id, const Hypernode& parent, LogWeight weight,
      SparseLogVec* fv = 0) :
    _id(id), _parent(parent), _weight(weight), _fv(fv), _label(-1) { }
  
  Hyperedge(int id, const Hypernode& parent,
      std::list<const Hypernode*> children, LogWeight weight,
      SparseLogVec* fv = 0) :
    _id(id), _parent(parent), _children(children), _weight(weight), _fv(fv),
    _label(-1) { }
  
  int getId() const { return _id; }
  
  const Hypernode& getParent() const { return _parent; }
  
  void setWeight(LogWeight weight) { _weight = weight; }
  
  LogWeight getWeight() const { return _weight; }
  
  const std::list<const Hypernode*>& getChildren() const { return _children; }
  
  const SparseLogVec* getFeatureVector() const { return _fv.get(); }
  
  int getLabel() const { return _label; }
  
  void setLabel(int label) { _label = label; }
  
private:
  int _id;
  const Hypernode& _parent;
  std::list<const Hypernode*> _children;
  LogWeight _weight;
  boost::shared_ptr<SparseLogVec> _fv;
  int _label;
};

#endif
