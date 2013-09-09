/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Kenneth Dwyer
 */

#ifndef _PHRASEPAIRSREADER_H
#define _PHRASEPAIRSREADER_H

#include "CognatePairReader.h"
#include "Label.h"
#include <string>
class Pattern;

class PhrasePairsReader : public CognatePairReader {

  public:
  
    virtual void readExample(const std::string& line, Pattern*& pattern,
        Label& label) const;
        
    static const std::string& name() {
      static const std::string _name = "PhrasePairs";
      return _name;
    }
};

#endif
