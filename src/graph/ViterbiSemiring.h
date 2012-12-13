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
#include "Ublas.h"
#include <algorithm>

class ViterbiSemiring {

public:
  ViterbiSemiring() : _score(LogWeight(0)) { }

  ViterbiSemiring(LogWeight score) : _score(score) { }
  
  ViterbiSemiring(const Hyperedge& edge) : _score(edge.getWeight()) { }
  
  LogWeight score() const {
    return _score;
  }
  
  ViterbiSemiring& operator+=(const ViterbiSemiring& toAdd) {
    _score = std::max(_score, toAdd.score());
    return (*this);
  }
  
  ViterbiSemiring& operator*=(const ViterbiSemiring& toProd) {
    _score *= toProd.score();
    return (*this);
  }
  
  static ViterbiSemiring one(const size_t numFeatures) {
    return ViterbiSemiring(LogWeight(1));
  }

  static ViterbiSemiring zero(const size_t numFeatures) {
    return ViterbiSemiring(LogWeight(0));
  }
  
  // Although insideOutside() cannot be called with this semiring, we need to
  // define this type in order to get everything to compile.
  typedef char InsideOutsideResult;
  
private:
  LogWeight _score;
};

#endif
