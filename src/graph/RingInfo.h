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

#include "FeatureVector.h"
#include "Hyperedge.h"
#include "LogWeight.h"
#include "Ring.h"
#include <algorithm>
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
using std::max;

class RingInfo {

private:
  // Either edge score (on edge), or accumulated mass (on state)
  LogWeight _score;
  
  // Either edge features (on edge), or accumulated features (on state)
  shared_ptr<FeatureVector<LogWeight> > _fv;

public:
  RingInfo() : _score(0) { }
  
  RingInfo(double score) : _score(score) { }
  
  RingInfo(double score, const FeatureVector<LogWeight>* fv) : _score(score),
      _fv(new FeatureVector<LogWeight>(*fv)) { }
  
  /**
   * Reinterprets an edgeInfo object with appropriate values for use in an edge representation
   * in the provided semi-ring
   * @param ei  Source edgeInfo
   * @param r   Semiring of interest
   */
  RingInfo(const Hyperedge& edge, Ring r) : _score(edge.getWeight()) {
    const FeatureVector<RealWeight>* fv = edge.getFeatureVector();
    const int d = fv->getNumEntries();
    shared_array<LogWeight> values(new LogWeight[d]);
    assert(_fv == 0);
    _fv.reset(new FeatureVector<LogWeight>(fvConvert(*fv, values, d)));
    
    if (r == RingExpectation)
      _fv->timesEquals(_score); // pese 
  }
  
  LogWeight score() const {
    return _score;
  }
  
  const shared_ptr<FeatureVector<LogWeight> > fv() const {
    return _fv;
  }

  /**
   * Semiring culmulative sum (Sum=LogAdd, Vit=Max)
   * @param ringInfo  Object to be accumulated
   * @param ring    Ring to be used
   */
  void collectSum(const RingInfo& toAdd, const Ring& ring) {
    if (ring == RingLog)
      _score.plusEquals(toAdd.score());
    else if (ring == RingViterbi)
      _score = max(_score, toAdd.score());
    else if (ring == RingExpectation) {
      _score.plusEquals(toAdd.score());
      toAdd.fv()->addTo(*_fv);
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
      _score.timesEquals(toProd.score());
    else if (ring == RingExpectation) {
      // <p,r> = <p1p2, p1r2 + p2r1>
      
      // p1r2 + p2r1
      _fv->timesEquals(toProd.score()); //p2r1
      FeatureVector<LogWeight> r2(*toProd.fv());
      r2.timesEquals(_score); //p1r2
      r2.addTo(*_fv);
    
      // p1p2
      _score.timesEquals(toProd.score());
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
      return RingInfo(LogWeight::kOne);
    else {
      assert(ring == RingExpectation);
      return RingInfo(LogWeight::kOne, new FeatureVector<LogWeight>());
    }
  }

  /**
   * Returns the additive identity for this semiring
   * @param ring  Semiring of interest
   * @return    Additive identity (0)
   */
  static RingInfo zero(const Ring& ring) {
    if (ring == RingLog || ring == RingViterbi)
      return RingInfo(LogWeight::kZero);
    else {
      assert(ring == RingExpectation);
      return RingInfo(LogWeight::kZero, new FeatureVector<LogWeight>());
    }
  }
};

#endif
