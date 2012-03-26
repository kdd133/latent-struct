/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Label.h"
#include "Pattern.h"
#include "StringPair.h"
#include "SentencePairReader.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include <vector>
using namespace boost;
using namespace std;

/* Lines are of the form:
 * 1<tab>TOK=UNK;P1G=D;S1G=g TOK=UNK;P1G=r;S1G=e<tab>TOK=UNK;P1G=R;S1G=e TOK=UNK;P1G=d;S1G=e
 * i.e., each word is a list of indicator features
 */
void SentencePairReader::readExample(const string& line, Pattern*& pattern,
    Label& label) const {
  assert(line.size() > 0);
  assert(!_addBeginEndMarkers); // Not supported by sentence-based feature gens.
  
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
  for (it = sourceTokens.begin(); it != sourceTokens.end(); ++it)
    source.push_back(*it);
    
  Tokenizer targetTokens(targetStr, spaceSep);
  vector<string> target;
  for (it = targetTokens.begin(); it != targetTokens.end(); ++it)
    target.push_back(*it);
    
  pattern = new StringPair(source, target);
}
