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
#include "WordPairReader.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include <vector>
using namespace boost;
using namespace std;

/* Read integer-mapped characters and store them as strings, adding begin- and
 * end-of-word markers to each string.
 * Lines are of the form: 0<tab>8 3 21<tab>42 9 8 1
 *   i.e., label \t source \t target (minus the spaces)
 */
void WordPairReader::readExample(const string& line, Pattern*& pattern,
    Label& label) const {
  assert(line.size() > 0);
  
  typedef tokenizer<char_separator<char> > Tokenizer;
  char_separator<char> tabSep("\t");
  char_separator<char> spaceSep(" ");
  
  Tokenizer fields(line, tabSep);
  Tokenizer::const_iterator it = fields.begin();
  label = lexical_cast<size_t>(*it++);
  string sourceStr = *it++;
  string targetStr = *it++;
  assert(it == fields.end());
  
  Tokenizer sourceTokens(sourceStr, spaceSep);
  vector<string> source;
  if (_addBeginEndMarkers)
    source.push_back(FeatureGenConstants::BEGIN_CHAR);
  for (it = sourceTokens.begin(); it != sourceTokens.end(); ++it)
    source.push_back(*it);
  if (_addBeginEndMarkers)
    source.push_back(FeatureGenConstants::END_CHAR);
    
  Tokenizer targetTokens(targetStr, spaceSep);
  vector<string> target;
  if (_addBeginEndMarkers)
    target.push_back(FeatureGenConstants::BEGIN_CHAR);
  for (it = targetTokens.begin(); it != targetTokens.end(); ++it)
    target.push_back(*it);
  if (_addBeginEndMarkers)
    target.push_back(FeatureGenConstants::END_CHAR);

  pattern = new StringPair(source, target);
}
