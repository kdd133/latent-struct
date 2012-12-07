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

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "Hyperedge.h"
#include "LogWeight.h"
#include "Ring.h"
#include <algorithm>
#include <boost/numeric/ublas/vector_sparse.hpp>

class RingInfo {

public:
  typedef boost::numeric::ublas::compressed_vector<LogWeight> sparse_vector;
  
  RingInfo() : _score(0) { }
  
  RingInfo(LogWeight score) : _score(score) { }
  
  RingInfo(LogWeight score, const FeatureVector<LogWeight>* fv) :
    _score(score) {
    if (fv)
      fv->toSparseVector(_fv);
  }
  
  /**
   * Reinterprets an edgeInfo object with appropriate values for use in an edge representation
   * in the provided semi-ring
   * @param ei  Source edgeInfo
   * @param r   Semiring of interest
   */
  RingInfo(const Hyperedge& edge, Ring r) : _score(edge.getWeight()) {
    _fv = *edge.getFeatureVector();
    
    if (r == RingExpectation)
      _fv *= _score; // pese 
  }
  
  LogWeight score() const {
    return _score;
  }
  
  const sparse_vector& fv() const {
    return _fv;
  }

  /**
   * Semiring culmulative sum (Sum=LogAdd, Vit=Max)
   * @param ringInfo  Object to be accumulated
   * @param ring    Ring to be used
   */
  void collectSum(const RingInfo& toAdd, const Ring& ring) {
    if (ring == RingLog)
      _score += toAdd.score();
    else if (ring == RingViterbi)
      _score = std::max(_score, toAdd.score());
    else if (ring == RingExpectation) {
      // <p,r> = <p1+p2, r1+r2>
      _score += toAdd.score();
      _fv += toAdd.fv();
    }
    else
      assert(0);
  }
  
  /**
   * Semiring culmulative product (Sum,Vit=Add)
   * @param riPath  Object to be accumulated
   * @param ring    Ring to be used
   */
  void collectProd(const RingInfo& toProd, const Ring& ring) {
    if (ring == RingLog || ring == RingViterbi)
      _score *= toProd.score();
    else if (ring == RingExpectation) {
      // <p,r> = <p1p2, p1r2 + p2r1>
      
      // p1r2 + p2r1
      _fv *= toProd.score(); // p2r1
      sparse_vector r2(toProd.fv());
      r2 *= _score; // p1r2
      _fv += r2;
    
      // p1p2
      _score *= toProd.score();
    }
    else
      assert(0);
  }
  
  /**
   * Returns the multiplicative identity for this semiring
   * @param ring  Semiring of interest
   * @return    Multiplicative identity (1)
   */
  static RingInfo one(const Ring& ring) {
    if (ring == RingLog || ring == RingViterbi)
      return RingInfo(LogWeight(1));
    else {
      assert(ring == RingExpectation);
      return RingInfo(LogWeight(1), new FeatureVector<LogWeight>());
    }
  }

  /**
   * Returns the additive identity for this semiring
   * @param ring  Semiring of interest
   * @return    Additive identity (0)
   */
  static RingInfo zero(const Ring& ring) {
    if (ring == RingLog || ring == RingViterbi)
      return RingInfo(LogWeight(0));
    else {
      assert(ring == RingExpectation);
      return RingInfo(LogWeight(0), new FeatureVector<LogWeight>());
    }
  }
  
private:
  // Either edge score (on edge), or accumulated mass (on state)
  LogWeight _score;
  
  // Either edge features (on edge), or accumulated features (on state)
  sparse_vector _fv;
};

#endif
