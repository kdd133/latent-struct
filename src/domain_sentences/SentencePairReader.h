/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _SENTENCEPAIRREADER_H
#define _SENTENCEPAIRREADER_H

#include "InputReader.h"
#include "Label.h"
#include <string>
using namespace std;
class Pattern;

class SentencePairReader : public InputReader {

  public:
  
    virtual void readExample(const string& line, Pattern*& pattern,
        Label& label) const;
        
    static const string& name() {
      static const string _name = "SentencePair";
      return _name;
    }
};

#endif
