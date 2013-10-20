/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _VITERBISEMIRING_H
#define _VITERBISEMIRING_H

#include "Hyperedge.h"
#include "LogWeight.h"

class ViterbiSemiring {

public:
  ViterbiSemiring() : _score(LogWeight()), _bp(0), _from(0) { }

  ViterbiSemiring(LogWeight score) : _score(score), _bp(0), _from(0) { }
  
  ViterbiSemiring(const Hyperedge& edge) : _score(edge.getWeight()),
      _bp(&edge), _from(0) { }
  
  LogWeight score() const {
    return _score;
  }
  
  const Hyperedge* bp() const {
    return _bp;
  }
  
  const ViterbiSemiring* from() const {
    return _from;
  }
  
  ViterbiSemiring& operator+=(const ViterbiSemiring& rhs) {
    if (!_bp || rhs._score > _score) {
      _score = rhs._score;
      _bp = rhs._bp;
      _from = rhs._from;
    }
    assert(_bp); // cannot be 0 following a += update
    return (*this);
  }
  
  ViterbiSemiring& operator*=(const ViterbiSemiring& rhs) {
    _score *= rhs._score;
    _from = &rhs;
    return (*this);
  }
  
  static ViterbiSemiring one(const size_t numFeatures) {
    return ViterbiSemiring(LogWeight(1));
  }

  static ViterbiSemiring zero(const size_t numFeatures) {
    return ViterbiSemiring(LogWeight());
  }
  
  bool operator<(const ViterbiSemiring& r) const {
    return _score >= r._score;
  }
  
  // Although insideOutside() cannot be called with this semiring, we need to
  // define this type in order to get everything to compile.
  typedef char InsideOutsideResult;
  
private:
  LogWeight _score;
  
  const Hyperedge* _bp; // back-pointer
  
  const ViterbiSemiring* _from; // predecessor in the chart (used by k-best)
};

#endif
