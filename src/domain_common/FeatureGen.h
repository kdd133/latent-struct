/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _FEATUREGEN_H
#define _FEATUREGEN_H

#include "Alphabet.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

class FeatureGen {
  
  public:
  
    FeatureGen(boost::shared_ptr<Alphabet> alphabet) :
      _alphabet(alphabet) {}
      
    virtual ~FeatureGen() {}
    
    virtual int processOptions(int argc, char** argv) = 0;
    
    void setAlphabet(boost::shared_ptr<Alphabet> alphabet) {
      _alphabet = alphabet;
    }
    
    boost::shared_ptr<Alphabet> getAlphabet() const {
      return _alphabet;
    }
    
    virtual bool isEmpty() {
      // "Empty" feature generators that do not produce any features must
      // override this and return true.
      return false;
    }
    
  protected:
  
    boost::shared_ptr<Alphabet> _alphabet;
};

#endif
