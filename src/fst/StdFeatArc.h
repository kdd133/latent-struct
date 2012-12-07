/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _STDFEATARC_H
#define _STDFEATARC_H

#include <fst/float-weight.h>
#include "FeatureVector.h"
#include "Ublas.h"

class RealWeight;

// Based on ArcTpl in "fst/arc.h".
class StdFeatArc {
  public:
    typedef fst::TropicalWeightTpl<double> Weight;
    typedef int Label;
    typedef int StateId;

    StdFeatArc(Label i, Label o, const Weight& w, StateId s,
        const SparseRealVec* f = 0)
      : ilabel(i), olabel(o), weight(w), nextstate(s), fv(f) { }
      
    StdFeatArc() : fv(0) { }

    static const string& Type() {
      static const string type = "standard";
      return type;
    }
  
    Label ilabel;
    Label olabel;
    Weight weight;
    StateId nextstate;
    const SparseRealVec* fv;
};

#endif
