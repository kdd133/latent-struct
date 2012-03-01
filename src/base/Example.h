/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _EXAMPLE_H
#define _EXAMPLE_H

#include <boost/shared_ptr.hpp>
using boost::shared_ptr;

#include "Label.h"
#include "Pattern.h"

class Example {

  public:
  
    Example(Pattern* pattern, Label label) : _y(label) {
      _x.reset(pattern);
    }
    
    Example() : _y(0) {
      _x.reset();
    }

    const Pattern* x() const { return _x.get(); }
    
    Label y() const { return _y; }

  private:
  
    shared_ptr<Pattern> _x;
    
    Label _y;

};

#endif
