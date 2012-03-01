/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _PATTERN_H
#define _PATTERN_H

#include <stddef.h>

class Pattern {

  public:
  
    Pattern() : _id(0) {}
    
    Pattern(size_t id) : _id(id) {}
    
    virtual ~Pattern() {}
    
    virtual int getSize() const = 0;
    
    size_t getId() const;
    
    void setId(size_t id);

  private:
  
    size_t _id;
};

inline size_t Pattern::getId() const {
  return _id;
}

inline void Pattern::setId(size_t id) {
  _id = id;
}

#endif
