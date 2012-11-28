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

#include "FeatureVector.h"
#include "LogWeight.h"
#include <boost/shared_ptr.hpp>
#include <list>
using boost::shared_ptr;
using std::list;

class Hypernode;
class RealWeight;

class Hyperedge {

public:
  Hyperedge(int id, const Hypernode& parent, LogWeight weight,
      FeatureVector<RealWeight>* fv = 0) :
    _id(id), _parent(parent), _weight(weight), _fv(fv) { }
  
  Hyperedge(int id, const Hypernode& parent, list<const Hypernode*> children,
      LogWeight weight, FeatureVector<RealWeight>* fv = 0) :
    _id(id), _parent(parent), _children(children), _weight(weight), _fv(fv) { }
  
  int getId() const;
  
  const Hypernode& getParent() const;
  
  void setWeight(LogWeight weight);
  
  LogWeight getWeight() const;
  
  const list<const Hypernode*>& getChildren() const;
  
  const FeatureVector<RealWeight>* getFeatureVector() const;
  
private:
  int _id;
  const Hypernode& _parent;
  list<const Hypernode*> _children;
  LogWeight _weight;
  shared_ptr<FeatureVector<RealWeight> > _fv;
  
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

inline const list<const Hypernode*>& Hyperedge::getChildren() const {
  return _children;
}

inline const FeatureVector<RealWeight>* Hyperedge::getFeatureVector() const {
  return _fv.get();
}

#endif
