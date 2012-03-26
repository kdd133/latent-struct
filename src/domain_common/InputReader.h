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

class InputReader {

  public:
  
    InputReader(bool addBeginEndMarkers = false) :
      _addBeginEndMarkers(addBeginEndMarkers) {}
  
    virtual ~InputReader() {}
  
    virtual void readExample(const std::string& line, Pattern*& pattern,
        Label& label) const = 0;
        
    void setAddBeginEndMarkers(bool status) {
      _addBeginEndMarkers = status;
    }
        
  protected:
  
    bool _addBeginEndMarkers;
};

#endif
