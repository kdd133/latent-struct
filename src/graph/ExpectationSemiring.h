/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _EXPECTATIONSEMIRING_H
#define _EXPECTATIONSEMIRING_H

#include "Hyperedge.h"
#include "LogWeight.h"
#include "Ublas.h"


class ExpectationSemiring {

public:

  ExpectationSemiring() : _score(LogWeight()) { }

  ExpectationSemiring(LogWeight score) : _score(score) { }
  
  ExpectationSemiring(LogWeight score, const SparseLogVec& fv) : _score(score),
      _fv(fv) {}
  
  ExpectationSemiring(const Hyperedge& edge) :
      _score(edge.getWeight()), _fv(*edge.getFeatureVector()) {
    _fv *= _score; // pese 
  }
  
  LogWeight score() const {
    return _score;
  }
  
  const SparseLogVec& fv() const {
    return _fv;
  }
  
  ExpectationSemiring& operator+=(const ExpectationSemiring& toAdd) {
    // <p,r> = <p1+p2, r1+r2>
    _score += toAdd.score();
    _fv += toAdd.fv();
    return (*this);
  }
  
  ExpectationSemiring& operator*=(const ExpectationSemiring& toProd) {
    // <p,r> = <p1p2, p1r2 + p2r1>
    _fv = _score * toProd.fv() + toProd.score() * _fv; // p1r2 + p2r1
    _score *= toProd.score(); // p1p2
    return (*this);
  }
  
  static ExpectationSemiring one(const size_t numFeatures) {
    return ExpectationSemiring(LogWeight(1), SparseLogVec(numFeatures));
  }

  static ExpectationSemiring zero(const size_t numFeatures) {
    return ExpectationSemiring(LogWeight(), SparseLogVec(numFeatures));
  }
  
  typedef struct {
    LogWeight Z;
    SparseLogVec* rBar;
    SparseLogVec* sBar;
    AccumLogMat* tBar;
  } InsideOutsideResult;
  
  static void initInsideOutsideAccumulator(const Graph& g,
      InsideOutsideResult& result) {
    result.rBar->clear();
//  result.sBar = 0; // currently unused
    result.tBar->clear();
    assert(result.rBar->size() == g.numFeatures());
    assert(result.tBar->size1() == g.numFeatures());
    assert(result.tBar->size2() == g.numFeatures());
  }
  
  static void accumulate(InsideOutsideResult& x,
      const ExpectationSemiring& keBar, const Hyperedge& e) {
    // In our applications, we have re == se
    const SparseLogVec& re = *e.getFeatureVector();
    const SparseLogVec& se = re;
    const SparseLogVec& keBarR = keBar.fv();
    const LogWeight pe = e.getWeight();
    const LogWeight keBarP = keBar.score();
    
    SparseLogVec pe_se = pe * se;
    
    ublas_util::addOuterProductLowerTriangular(re, se, keBarP * pe, *x.tBar);
    ublas_util::addOuterProductLowerTriangular(pe_se, keBarR,
        LogWeight(0, true), *x.tBar);
    pe_se *= keBarP;
    *x.rBar += pe_se;
  }
  
  static void finalizeInsideOutsideResult(InsideOutsideResult& result,
      const ExpectationSemiring& betaRoot) {
    // rBar and tBar have (presumably) already been set via accumulation
    result.Z = betaRoot.score();
//  result.sBar = betaRoot.fv(); // currently not used, so don't bother copying
  }

private:
  LogWeight _score;
  
  SparseLogVec _fv;
};

#endif
