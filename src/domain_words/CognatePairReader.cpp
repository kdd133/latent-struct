/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "FeatureGenConstants.h"
#include "Label.h"
#include "Pattern.h"
#include "StringPair.h"
#include "CognatePairReader.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

/* Read sequences of ISO-8859 encoded character and store them as strings,
 * adding begin- and end-of-word markers to each string.
 * Lines are of the form: {+,-} drehende freshened
 */
void CognatePairReader::readExample(const string& line, Pattern*& pattern,
    Label& label) const {
  assert(line.size() > 0);
  
  typedef tokenizer<char_separator<char> > Tokenizer;
  char_separator<char> spaceSep(" ");
  
  Tokenizer fields(line, spaceSep);
  Tokenizer::const_iterator it = fields.begin();
  string signChar = *it++;
  assert(signChar == "+" || signChar == "-");
  label = (signChar == "+") ? 1 : 0;
  string sourceStr = *it++;
  string targetStr = *it++;
  assert(it == fields.end());
  
  vector<string> source;
  if (_addBeginEndMarkers)
    source.push_back(FeatureGenConstants::BEGIN_CHAR);
  for (size_t i = 0; i < sourceStr.size(); ++i) {
    string s;
    s.append(1, sourceStr[i]);
    source.push_back(s);
  }
  if (_addBeginEndMarkers)
    source.push_back(FeatureGenConstants::END_CHAR);

  vector<string> target;
  if (_addBeginEndMarkers)
    target.push_back(FeatureGenConstants::BEGIN_CHAR);
  for (size_t i = 0; i < targetStr.size(); ++i) {
    string t;
    t.append(1, targetStr[i]);
    target.push_back(t);
  }
  if (_addBeginEndMarkers)
    target.push_back(FeatureGenConstants::END_CHAR);
    
  pattern = new StringPair(source, target);
}
