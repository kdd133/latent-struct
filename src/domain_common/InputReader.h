/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _INPUTREADER_H
#define _INPUTREADER_H

#include <string>
using std::string;

class InputReader {

  public:
  
    virtual ~InputReader() {}
  
    virtual void readExample(const string& line, Pattern*& pattern,
        Label& label) const = 0;
};

#endif
