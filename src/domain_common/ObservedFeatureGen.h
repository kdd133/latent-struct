/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OBSERVEDFEATUREGEN_H
#define _OBSERVEDFEATUREGEN_H

#include "FeatureGen.h"
#include "FeatureVector.h"
#include "Label.h"
#include <boost/shared_ptr.hpp>

class Pattern;
class RealWeight;

class ObservedFeatureGen : public FeatureGen {

  public:
  
    ObservedFeatureGen(boost::shared_ptr<Alphabet> a) : FeatureGen(a) {}

    virtual ~ObservedFeatureGen() {}

    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
        const Label y) = 0;
  
    virtual int processOptions(int argc, char** argv) = 0;
};

#endif
