/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _EMPTYOBSERVEDFEATUREGEN_H
#define _EMPTYOBSERVEDFEATUREGEN_H

#include "FeatureVector.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include <boost/shared_ptr.hpp>

class Alphabet;
class Pattern;
class RealWeight;


class EmptyObservedFeatureGen : public ObservedFeatureGen {

  public:
  
    // TODO: Other classes derived from ObservedFeatureGen take an Alphabet
    // in their constructors, so we're doing this just to maintain
    // "polymorphism" in latent_struct.cpp. Maybe this situation should be
    // cleaned up so that any subclass of ObservedFeatureGen is forced to
    // provide such a constructor.
    EmptyObservedFeatureGen(boost::shared_ptr<Alphabet> alphabet) :
      ObservedFeatureGen(alphabet) {}
    
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
        const Label y) {
      return new FeatureVector<RealWeight>(); // return the zero vector
    }
    
    virtual int processOptions(int argc, char** argv) {
      return 0;
    }
    
    static const string& name() {
      static const string _name = "Empty";
      return _name;
    }

};

#endif
