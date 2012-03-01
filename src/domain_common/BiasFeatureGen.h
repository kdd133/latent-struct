/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _BIASFEATUREGEN_H
#define _BIASFEATUREGEN_H

#include "Alphabet.h"
#include "FeatureVector.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include <boost/shared_ptr.hpp>
#include <string>
using std::string;

class Pattern;
class RealWeight;


class BiasFeatureGen : public ObservedFeatureGen {

  public:
  
    BiasFeatureGen(boost::shared_ptr<Alphabet> alphabet, bool normalize = true);
    
    virtual FeatureVector<RealWeight>* getFeatures(const Pattern& x,
        const Label y);
    
    virtual int processOptions(int argc, char** argv);
    
    static const string& name() {
      static const string _name = "Bias";
      return _name;
    }
    
    static const string kPrefix;

  private:
  
    bool _normalize;
    
};

#endif
