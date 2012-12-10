/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _RINGINFO_H
#define _RINGINFO_H

#include "Hyperedge.h"
#include "LogWeight.h"
#include "Ring.h"
#include "Ublas.h"
#include <algorithm>
#include <boost/numeric/ublas/vector_sparse.hpp>

class RingInfo {

public:
  RingInfo() : _ring(RingLog), _score(LogWeight(0)) { }

  RingInfo(Ring ring) : _ring(ring), _score(LogWeight(0)) { }
  
  RingInfo(Ring ring, LogWeight score) : _ring(ring), _score(score) { }
  
  RingInfo(Ring ring, LogWeight score, const SparseLogVec& fv) : _ring(ring),
      _score(score), _fv(fv) {}
  
  RingInfo(Ring ring, const Hyperedge& edge) : _ring(ring),
      _score(edge.getWeight()) {
    _fv = *edge.getFeatureVector();    
    if (_ring == RingExpectation)
      _fv *= _score; // pese 
  }
  
  LogWeight score() const {
    return _score;
  }
  
  const SparseLogVec& fv() const {
    return _fv;
  }
  
  /**
   * Semiring culmulative sum (Sum=LogAdd, Vit=Max)
   * @param ringInfo  Object to be accumulated
   * @param ring    Ring to be used
   */
  void collectSum(const RingInfo& toAdd) {
    if (_ring == RingLog)
      _score += toAdd.score();
    else if (_ring == RingViterbi)
      _score = std::max(_score, toAdd.score());
    else if (_ring == RingExpectation) {
      // <p,r> = <p1+p2, r1+r2>
      _score += toAdd.score();
      _fv += toAdd.fv();
    }
    else
      assert(0);
  }
  
  void collectProd(const RingInfo& toProd) {
    if (_ring == RingLog || _ring == RingViterbi)
      _score *= toProd.score();
    else if (_ring == RingExpectation) {
      // <p,r> = <p1p2, p1r2 + p2r1>
      
      // p1r2 + p2r1
      _fv *= toProd.score(); // p2r1
      SparseLogVec r2(toProd.fv());
      r2 *= _score; // p1r2
      _fv += r2;
    
      // p1p2
      _score *= toProd.score();
    }
    else
      assert(0);
  }
  
  static RingInfo one(const Ring& ring, const size_t numFeatures) {
    RingInfo r(ring, LogWeight(1));
    if (ring == RingLog || ring == RingViterbi)
      return r;
    else {
      assert(ring == RingExpectation);
      r._fv = SparseLogVec(numFeatures);
      return r;
    }
  }

  static RingInfo zero(const Ring& ring, const size_t numFeatures) {
    RingInfo r(ring, LogWeight(0));
    if (ring == RingLog || ring == RingViterbi)
      return r;
    else {
      assert(ring == RingExpectation);
      r._fv = SparseLogVec(numFeatures);
      return r;
    }
  }
  
private:
  Ring _ring;
  
  LogWeight _score;
  
  SparseLogVec _fv;
};

#endif
