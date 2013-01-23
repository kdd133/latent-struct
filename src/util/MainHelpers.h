/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _MAINHELPERS_H
#define _MAINHELPERS_H

#include "Dataset.h"
#include "Parameters.h"
#include <set>
#include <string>

class AlignmentFeatureGen;
class Alphabet;
class TrainingObjective;
class WeightVector;

// Classify eval examples and optionally write the predictions to files.
// Can also write the alignments to files upon request.
void evaluateMultipleWeightVectors(const std::vector<Parameters>&,
    const Dataset&, TrainingObjective&, const std::string&, int id,
    bool writeFiles, bool writeAlignments, bool cachingEnabled);

void initWeights(WeightVector& w, const std::string& initType, double noise,
    int seed, const boost::shared_ptr<Alphabet> alphabet,
    const std::set<Label>& labels,
    const boost::shared_ptr<const AlignmentFeatureGen> fgen);

#endif
