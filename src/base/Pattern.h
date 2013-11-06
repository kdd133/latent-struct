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
#include <string>


class Pattern {

  public:
  
    Pattern() : _id(0) {}
    
    Pattern(std::size_t id) : _id(id) {}
    
    virtual ~Pattern() {}
    
    virtual int getSize() const = 0;
    
    std::size_t getId() const;
    
    void setId(std::size_t id);
    
    virtual std::string getHashString() const = 0;

  private:
  
    std::size_t _id;
};

inline std::size_t Pattern::getId() const {
  return _id;
}

inline void Pattern::setId(std::size_t id) {
  _id = id;
}

#endif
