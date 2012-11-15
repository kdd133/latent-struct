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
#include <list>
using namespace boost;
using namespace std;

class Hypernode;
class RealWeight;

class Hyperedge {

public:
  Hyperedge(int id, const Hypernode& parent, double weight,
      FeatureVector<RealWeight>* fv = 0) :
    _id(id), _parent(parent), _weight(weight), _fv(fv) { }
  
  Hyperedge(int id, const Hypernode& parent, list<const Hypernode*> children,
      double weight, FeatureVector<RealWeight>* fv = 0) :
    _id(id), _parent(parent), _children(children), _weight(weight), _fv(fv) { }
  
  int getId() const;
  
  const Hypernode& getParent() const;
  
  void setWeight(double weight);
  
  double getWeight() const;
  
  const list<const Hypernode*>& getChildren() const;
  
  const FeatureVector<RealWeight>* getFeatureVector() const;
  
private:
  int _id;
  const Hypernode& _parent;
  list<const Hypernode*> _children;
  double _weight;
  shared_ptr<FeatureVector<RealWeight> > _fv;
  
};

inline int Hyperedge::getId() const {
  return _id;
}

inline const Hypernode& Hyperedge::getParent() const {
  return _parent;
}

inline void Hyperedge::setWeight(double weight) {
  _weight = weight;
}

inline double Hyperedge::getWeight() const {
  return _weight;
}

inline const list<const Hypernode*>& Hyperedge::getChildren() const {
  return _children;
}

inline const FeatureVector<RealWeight>* Hyperedge::getFeatureVector() const {
  return _fv.get();
}

#endif
