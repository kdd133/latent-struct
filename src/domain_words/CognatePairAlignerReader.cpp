/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "CognatePairAlignerReader.h"
#include "FeatureGenConstants.h"
#include "Label.h"
#include "Pattern.h"
#include "StringPair.h"
#include "StringPairAligned.h"
#include <assert.h>
#include <string>
#include <vector>

using namespace std;

void CognatePairAlignerReader::readExample(const string& line, Pattern*& pattern,
    Label& label) const {
  Pattern* pat = 0;
  CognatePairReader::readExample(line, pat, label);  
  StringPair* sp = (StringPair*)pat;
  pattern = new StringPairAligned(sp->getSource(), sp->getTarget());
  delete sp;
}
