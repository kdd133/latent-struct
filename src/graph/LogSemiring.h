/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LOGSEMIRING_H
#define _LOGSEMIRING_H

#include "Hyperedge.h"
#include "LogWeight.h"
#include "Ublas.h"

class LogSemiring {

public:
  LogSemiring() : _score(LogWeight()) { }
  
  LogSemiring(LogWeight score) : _score(score) { }
  
  LogSemiring(const Hyperedge& edge) : _score(edge.getWeight()) { }
  
  LogWeight score() const {
    return _score;
  }
  
  LogSemiring& operator+=(const LogSemiring& toAdd) {
    _score += toAdd.score();
    return (*this);
  }
  
  LogSemiring& operator*=(const LogSemiring& toProd) {
    _score *= toProd.score();
    return (*this);
  }
  
  static LogSemiring one(const size_t numFeatures) {
    return LogSemiring(LogWeight(1));
  }

  static LogSemiring zero(const size_t numFeatures) {
    return LogSemiring(LogWeight());
  }

  typedef struct {
    LogWeight Z;
    SparseLogVec* rBar;
  } InsideOutsideResult;

  static void initInsideOutsideAccumulator(const Graph& g,
      InsideOutsideResult& result) {
    result.rBar->clear();
    assert(result.rBar->size() == g.numFeatures());
  }

  static void accumulate(InsideOutsideResult& x, const LogSemiring& keBar,
      const Hyperedge& e) {
    // Compute: xHat = xHat + keBar*xe
    //    where xe = pe*re
    SparseLogVec pe_re = *e.getFeatureVector();
    pe_re *= e.getWeight() * keBar.score();
    *x.rBar += pe_re;
  }
  
  static void finalizeInsideOutsideResult(InsideOutsideResult& result,
      const LogSemiring& betaRoot) {
    // rBar has (presumably) already been set via accumulation
    result.Z = betaRoot.score();
  }

private:
  LogWeight _score;
};

#endif
