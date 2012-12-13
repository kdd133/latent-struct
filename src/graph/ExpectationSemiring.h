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
  ExpectationSemiring() : _score(LogWeight(0)) { }

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
    
    // p1r2 + p2r1
    _fv *= toProd.score(); // p2r1
    SparseLogVec r2(toProd.fv());
    r2 *= _score; // p1r2
    _fv += r2;
  
    // p1p2
    _score *= toProd.score();
    
    return (*this);
  }
  
  static ExpectationSemiring one(const size_t numFeatures) {
    return ExpectationSemiring(LogWeight(1), SparseLogVec(numFeatures));
  }

  static ExpectationSemiring zero(const size_t numFeatures) {
    return ExpectationSemiring(LogWeight(0), SparseLogVec(numFeatures));
  }
  
  typedef struct {
    LogVec rBar;
    LogMat tBar;
  } accumulator;
  
  typedef struct {
    LogWeight Z;
    LogVec rBar;
    LogVec sBar;
    LogMat tBar;
  } InsideOutsideResult;
  
  static void initInsideOutsideAccumulator(const std::size_t d,
      InsideOutsideResult& result) {
    result.rBar.resize(d);
    result.tBar.resize(d, d);
  }
  
  static void accumulate(InsideOutsideResult& x,
      const ExpectationSemiring& keBar, const Hyperedge& e) {
    const LogWeight pe = e.getWeight();
      
    // In our applications, we have re == se
    SparseLogVec re = *e.getFeatureVector();
    SparseLogVec& se = re;
    
    SparseLogMat pe_re_se = outer_prod(re, se);
    pe_re_se *= pe;
    
    const SparseLogVec& keBarR = keBar.fv();
    
    SparseLogVec& pe_se = se;
    pe_se *= pe;
    const SparseLogMat pe_se_keBarR = outer_prod(pe_se, keBarR);
    
    const LogWeight& keBarP = keBar.score();
    
    pe_se *= keBarP;    // = keBarP * pe_se
    pe_re_se *= keBarP; // = keBarP * pe_re_se
    
    x.rBar += pe_se;
    x.tBar += (pe_re_se + pe_se_keBarR);
  }
  
  static void finalizeInsideOutsideResult(InsideOutsideResult& result,
      const ExpectationSemiring& betaRoot) {
    // rBar and tBar have (presumably) already been set via accumulation
    result.Z = betaRoot.score();
    result.sBar = betaRoot.fv();
  }
  
private:
  LogWeight _score;
  
  SparseLogVec _fv;
};

#endif
