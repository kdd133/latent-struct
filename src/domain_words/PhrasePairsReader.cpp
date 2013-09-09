/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Kenneth Dwyer
 */

#include "CognatePairReader.h"
#include "FeatureGenConstants.h"
#include "Label.h"
#include "Pattern.h"
#include "StringPairPhrases.h"
#include "PhrasePairsReader.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

/* Each line contains a set of phrase pairs, which are separated by pipe
 * symbols. The first "field" of a line contains the class label {+,-} and the
 * source-target pair; it is separated from the phrases by a tab symbol.
 * For example, the aligned pair
 *                          ^ o t v e r t k a $
 *                          ^ o - v e r t - - $
 * corresponds to the following line:
 * - otvertka overt<tab>rt rt$|er ert|^o ^|ert er|tv ov|rtk r| ... v v|r rt|
 * The phrases are produced by Shane Bergsma's pairs2phrases.pl script, whose
 * output is post-processed to prepend labels and remove "--" lines, so that
 * there is one example per line as desired.
 * See also: src/scripts/pairs2phrases_postprocess.sh
 */
void PhrasePairsReader::readExample(const string& line, Pattern*& pattern,
    Label& label) const {
  assert(line.size() > 0);
  
  typedef tokenizer<char_separator<char> > Tokenizer;
  char_separator<char> pipeSep("|");
  char_separator<char> spaceSep(" ");
  char_separator<char> tabSep("\t");
  
  Tokenizer fields(line, tabSep);
  Tokenizer::const_iterator it = fields.begin();
  string labelSourceTarget = *it++;
  string phrases = *it++;
  
  Pattern* pat = 0;
  CognatePairReader::readExample(labelSourceTarget, pat, label);
  StringPair* sp = (StringPair*)pat;
  
  vector<string> phrasePairs;
  fields = Tokenizer(phrases, pipeSep);
  it = fields.begin();
  while (it != fields.end())
    phrasePairs.push_back(*it++);
  
  pattern = new StringPairPhrases(sp->getSource(), sp->getTarget(), phrasePairs);
}
