/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "FeatureGenConstants.h"
#include "NgramLexicon.h"
#include <boost/tokenizer.hpp>
#include <fstream>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

NgramLexicon::NgramLexicon(string fname) {
  ifstream fin(fname.c_str(), ifstream::in);
  if (!fin.good())
    throw "Could not open " + fname;

  char_separator<char> sep(" \t");
  string line;
  while (getline(fin, line)) {
    vector<string> token_vec;
    tokenizer<char_separator<char> > tokens(line, sep);
    tokenizer<char_separator<char> >::const_iterator it;
    for (it = tokens.begin(); it != tokens.end(); ++it) {
      const string t = *it;
      if (t == "<s>") // SRILM begin marker
        token_vec.push_back(FeatureGenConstants::BEGIN_CHAR);
      else if (t == "</s>") // SRILM end marker
        token_vec.push_back(FeatureGenConstants::END_CHAR);
      else
        token_vec.push_back(t);
    }

    // Note: the -1 is to omit the last token, which is the n-gram's frequency
    string ngram = getNgramString(token_vec, token_vec.size()-1);
    
    _ngrams.insert(ngram);
  }
  fin.close();
}

bool NgramLexicon::contains(const string& ngram) const {
  return _ngrams.find(ngram) != _ngrams.end();
}

string NgramLexicon::getNgramString(const vector<string>& ngram, int len,
    int start) {
  string ngram_str = ngram[start];
  for (size_t i = start+1; i < start+len && i < ngram.size(); ++i)
    ngram_str += " " + ngram[i];
  return ngram_str;
}
