/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _TRANSDUCER_H
#define _TRANSDUCER_H

#include "FeatureVector.h"
#include <boost/shared_array.hpp>

class LogWeight;
class RealWeight;

//Note: A fully observed model can use a transducer that deterministically maps
//a given string pattern and output label to a fixed feature vector.
class Transducer {
  public:
    virtual ~Transducer() {}
    
    virtual LogWeight logPartition() = 0;

    virtual LogWeight logExpectedFeaturesUnnorm(FeatureVector<LogWeight>& fv,
        boost::shared_array<LogWeight> buffer) = 0;

    virtual RealWeight maxFeatureVector(FeatureVector<RealWeight>& fv,
        bool getCostOnly = false) = 0;

};
#endif
