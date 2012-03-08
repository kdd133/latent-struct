/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _FEATUREGENCONSTANTS_H
#define _FEATUREGENCONSTANTS_H

class FeatureGenConstants {

  public:
  
    // Separator for components of a feature.
    static const char* PART_SEP;
    
    // Separator for elements of a phrase.
    static const char* PHRASE_SEP;
    
    // Represents an epsilon/gap in an alignment string.
    static const char* EPSILON;
    
    // Separator for edit operations, e.g., source > target.
    static const char* OP_SEP;
    
    // Separator for the indicator features in a word. 
    static const char* WORDFEAT_SEP;  
    
    // Beginning of string marker.
    static const char* BEGIN_CHAR;
    
    // End of string marker.
    static const char* END_CHAR;
};

#endif
