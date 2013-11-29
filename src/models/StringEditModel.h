/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _STRINGEDITMODEL_H
#define _STRINGEDITMODEL_H

#include "AlignmentFeatureGen.h"
#include "ExpectationSemiring.h"
#include "Hyperedge.h"
#include "Inference.h"
#include "InputReader.h"
#include "KBestViterbiSemiring.h"
#include "Label.h"
#include "LogSemiring.h"
#include "LogWeight.h"
#include "Model.h"
#include "NgramLexicon.h"
#include "ObservedFeatureGen.h"
#include "OpDelete.h"
#include "OpInsert.h"
#include "OpMatch.h"
#include "OpNone.h"
#include "OpReplace.h"
#include "OpSubstitute.h"
#include "Pattern.h"
#include "StateType.h"
#include "StringEditModel.h"
#include "StringPair.h"
#include "Ublas.h"
#include "ViterbiSemiring.h"
#include "WeightVector.h"
#include <algorithm>
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>


template <typename Graph>
class StringEditModel : public Model {

  public:
  
    StringEditModel(boost::shared_ptr<AlignmentFeatureGen> fgenAlign,
      boost::shared_ptr<ObservedFeatureGen> fgenObserved);
    
    ~StringEditModel() {};
    
    virtual int processOptions(int argc, char** argv);
    
    virtual size_t gatherFeatures(const Pattern& pattern,
      const Label label);
    
    virtual LogWeight totalMass(const WeightVector& w, const Pattern& pattern,
      const Label label);
    
    virtual double viterbiScore(const WeightVector& w, const Pattern& pattern,
      const Label label);
  
    virtual double maxFeatures(const WeightVector& w, SparseRealVec* fv,
      const Pattern& pattern, const Label label,
      bool includeObservedFeaturesArc = true);
    
    virtual LogWeight expectedFeatures(const WeightVector& w, SparseLogVec* fv,
      const Pattern& pattern, const Label label, bool normalize = true);
      
    virtual LogWeight expectedFeatureCooccurrences(
      const WeightVector& w, AccumLogMat* fm, SparseLogVec* fv,
      const Pattern& pattern, const Label label, bool normalize = true);
      
    virtual boost::shared_ptr<const SparseRealVec> observedFeatures(
      const Pattern& pattern, const Label label);
      
    virtual void getBestAlignments(std::ostream& alignmentStringRepresentations,
      boost::shared_ptr<std::vector<boost::shared_ptr<SparseRealVec> > >&
        maxFvForEachAlignment,
      const WeightVector& w, const Pattern& pattern, const Label label,
      bool observedFeatures = true);
    
    static const std::string& name() {
      static const std::string _name = "StringEdit";
      return _name;
    }
    
    virtual void emptyCache();
    
    typedef std::pair<std::string, Label> ExampleId;
    

  private:
  
    // If true, the graph we build will consider exact matches, in addition to
    // generic substitutions.
    bool _useMatch;
    
    // If true, the graph we build will allow a deletion to follow an insertion.
    bool _allowRedundant;
    
    // Maximum phrase length on the source side.
    int _maxSourcePhraseLength;
    
    // Maximum phrase length on the target side.
    int _maxTargetPhraseLength;
    
    // The Markov order -- how many previous states the current state may
    // depend on.
    int _order;
    
    // A list of state types that comprise the nodes in a graph.
    boost::ptr_vector<StateType> _states;
    
    // A list of the edit operations that are in use.
    boost::ptr_vector<EditOperation> _ops;
    
    // If true, do not fire any features for arc connecting to the final state.
    bool _noFinalArcFeats;
    
    // If true, the graph will include an arc that fires a feature for the start
    // state; this arc is included in all paths through the graph.
    bool _includeStartArc;
    
    // If true, omit features for pairs of distinct aligned characters. In other
    // words, disallow substitutions for unigrams. Insertions and deletions of
    // single characters are included regardless of this option.
    bool _identicalUnigramsOnly;

    // A cache for Fsts that include an arc for the observed features. 
    boost::ptr_map<ExampleId, Graph> _fstCache;
    
    // A cache for Fsts that omit the arc for the observed features.
    boost::ptr_map<ExampleId, Graph> _fstCacheNoObs;
    
    // A cache for observed feature vectors.
    boost::unordered_map<ExampleId, boost::shared_ptr<const SparseRealVec> >
      _fvCacheObs;
    
    // If caching is disabled, it's still useful to reuse an existing graph in
    // order to avoid repeated memory allocation.
    boost::shared_ptr<Graph> _reusableGraph;
    
    // If set to a non-zero value, approximate the feature co-occurrence counts
    // that are obtained in expectedFeatureCooccurrences() by drawing the
    // number of samples specified by this parameter.
    int _featureCoocSamples;    
    
    // If true, only cache observed feature vectors (regardless of the value of
    // _cacheGraphs, which is inherited from class Model).
    bool _cacheObsFeatsOnly;
    
    Graph* getGraph(boost::ptr_map<ExampleId, Graph>& cache,
        const WeightVector& w, const Pattern& x, const Label y,
        bool includeObsFeaturesArc = true);
        
    void alignmentsToMaxFvs(
        const std::vector<std::list<const Hyperedge*> >& alignments,
        boost::shared_ptr<std::vector<boost::shared_ptr<SparseRealVec> > >& maxFvs,
        int dim);
        
    void addZeroOrderStates();
    
    void addFirstOrderStates();
    
    void addSecondOrderStates();
    
    StringEditModel(const StringEditModel& x);
    StringEditModel& operator=(const StringEditModel& x);
};

template <typename Graph>
StringEditModel<Graph>::StringEditModel(
    boost::shared_ptr<AlignmentFeatureGen> fgenAlign,
    boost::shared_ptr<ObservedFeatureGen> fgenObserved) :
    Model(fgenAlign, fgenObserved), _useMatch(false), _allowRedundant(false),
    _maxSourcePhraseLength(1), _maxTargetPhraseLength(1), _order(1),
    _noFinalArcFeats(false), _includeStartArc(false),
    _cacheObsFeatsOnly(false) {
}

template <typename Graph>
void StringEditModel<Graph>::addZeroOrderStates() {
  using boost::lexical_cast;
  using std::string;
  
  _states.push_back(new StateType(0, "sta"));
  StateType& start = _states.front();
  
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    string opName = "Del" + lexical_cast<string>(s);
    EditOperation* op = new OpDelete(_ops.size(), &start, opName, s);
    _ops.push_back(op);
    start.addValidOperation(op);
  }
  
  for (int t = 1; t <= _maxTargetPhraseLength; t++) {
    string opName = "Ins" + lexical_cast<string>(t);
    EditOperation* op = new OpInsert(_ops.size(), &start, opName, t);
    _ops.push_back(op);
    start.addValidOperation(op);
  }
  
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    const string sStr = lexical_cast<string>(s);
    for (int t = 1; t <= _maxTargetPhraseLength; t++) {
      const string tStr = lexical_cast<string>(t);
      if (_useMatch) { // Distinguish between Substitutes and Matches.
        // If _identicalUnigramsOnly is true, omit Sub operations for pairs of
        // unigrams.
        if (!_identicalUnigramsOnly || s > 1 || t > 1) {
          string opName = "Sub" + sStr + tStr;
          EditOperation* op = new OpSubstitute(_ops.size(), &start, opName, s, t);
          _ops.push_back(op);
          start.addValidOperation(op);
        }
            
        if (s == t) { // can't possibly match phrases of different lengths
          string opName =  "Mat" + sStr + tStr;
          EditOperation* op = new OpMatch(_ops.size(), &start, opName, s);
          _ops.push_back(op);
          start.addValidOperation(op);
        }
      }
      else { // !_useMatch: Use Replace only, instead of Substitute and Match.
        string opName = "Rep" + sStr + tStr;
        EditOperation* op = new OpReplace(_ops.size(), &start, opName, s, t);
        _ops.push_back(op);
        start.addValidOperation(op);
      }
    }
  }
}

template <typename Graph>
void StringEditModel<Graph>::addFirstOrderStates() {
  using namespace boost;
  using namespace std;
  
  //// Add states ////
  
  StateType* start = new StateType(_states.size(), "sta");
  _states.push_back(start);
  
  vector<StateType*> ins(_maxTargetPhraseLength + 1); // We'll skip entry 0
  for (int t = 1; t <= _maxTargetPhraseLength; t++) {
    StateType* state = new StateType(_states.size(), "ins" +
        lexical_cast<string>(t));
    ins[t] = state;
    _states.push_back(state);
  }
  
  vector<StateType*> del(_maxSourcePhraseLength + 1); // We'll skip entry 0
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    StateType* state = new StateType(_states.size(), "del" +
        lexical_cast<string>(s));
    del[s] = state;
    _states.push_back(state);
  }
  
  vector<StateType*> mat;
  if (_useMatch) {
    // We can't possibly match phrases of different lengths, so the smaller of
    // the two maximum phrase lengths equals the longest possible match length.
    const int maxMatchLength = std::min(_maxSourcePhraseLength,
        _maxTargetPhraseLength);
    mat.resize(maxMatchLength + 1); // We'll skip entry 0
    for (int s = 1; s <= maxMatchLength; s++) {
      StateType* state = new StateType(_states.size(), "mat" +
          lexical_cast<string>(s));
      mat[s] = state;
      _states.push_back(state);
    }
  }
  
  typedef multi_array<StateType*, 2> array2d;
  array2d sub(extents[_maxSourcePhraseLength + 1][_maxTargetPhraseLength + 1]);
  array2d rep(extents[_maxSourcePhraseLength + 1][_maxTargetPhraseLength + 1]);
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    const string sStr = lexical_cast<string>(s);
    for (int t = 1; t <= _maxTargetPhraseLength; t++) {
      const string tStr = lexical_cast<string>(t);
      if (_useMatch) {
        if (!_identicalUnigramsOnly || s > 1 || t > 1) {
          StateType* state = new StateType(_states.size(), "sub" + sStr + tStr);
          sub[s][t] = state;
          _states.push_back(state);
        }
      }
      else {
        StateType* state = new StateType(_states.size(), "rep" + sStr + tStr);
        rep[s][t] = state;
        _states.push_back(state);
      }
    }
  }
  
  //// Add edit operations ////
  
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    string opName = "Del" + lexical_cast<string>(s);
    EditOperation* op = new OpDelete(_ops.size(), del[s], opName, s);
    _ops.push_back(op);
    BOOST_FOREACH(StateType& source, _states) {
      // Disallow ins->del if the --allow-redundant flag is absent.
      if (!_allowRedundant && starts_with(source.getName(), "ins"))
        continue;
      source.addValidOperation(op);
    }
  }
  
  for (int t = 1; t <= _maxTargetPhraseLength; t++) {
    string opName = "Ins" + lexical_cast<string>(t);
    EditOperation* op = new OpInsert(_ops.size(), ins[t], opName, t);
    _ops.push_back(op);
    BOOST_FOREACH(StateType& source, _states) {
      source.addValidOperation(op);
    }
  }
  
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    const string sStr = lexical_cast<string>(s);
    for (int t = 1; t <= _maxTargetPhraseLength; t++) {
      const string tStr = lexical_cast<string>(t);
      if (_useMatch) { // Distinguish between Substitutes and Matches.
        // If _identicalUnigramsOnly is true, omit Sub operations for pairs of
        // unigrams.
        if (!_identicalUnigramsOnly || s > 1 || t > 1) {
          string opName = "Sub" + sStr + tStr;
          EditOperation* op = new OpSubstitute(_ops.size(), sub[s][t], opName,
              s, t);
          _ops.push_back(op);
          BOOST_FOREACH(StateType& source, _states) {
            source.addValidOperation(op);
          }
        }

        if (s == t) { // can't possibly match phrases of different lengths
          string opName = "Mat" + sStr + tStr;
          EditOperation* op = new OpMatch(_ops.size(), mat[s], opName, s);
          _ops.push_back(op);
          BOOST_FOREACH(StateType& source, _states) {
            source.addValidOperation(op);
          }
        }
      }
      else { // !_useMatch: Use Replace only, instead of Substitute and Match.
        string opName = "Rep" + sStr + tStr;
        EditOperation* op = new OpReplace(_ops.size(), rep[s][t], opName, s, t);
        _ops.push_back(op);
        BOOST_FOREACH(StateType& source, _states) {
          source.addValidOperation(op);
        }
      }
    }
  }
}

template <typename Graph>
void StringEditModel<Graph>::addSecondOrderStates() {
  using namespace boost;
  using namespace std;
  
  // FIXME: Longer phrase lengths are not actually supported here yet, since we
  // have more operations than states; this means the history can be ambiguous.
  // That is, the states do not encode phrase lengths, so when looking at the
  // history one cannot know whether, e.g., the transition to state "sub" was
  // via the edit operation "Ins1" or "Ins2".
  assert(_maxSourcePhraseLength == 1 && _maxTargetPhraseLength == 1);
  
  // Note: Chosen so as not to overlap with anything in FeatureGenConstants.
  const string trans = "/";
  
  const string start = "sta";
  const string transToStart = trans + start;

  vector<string> baseEditNames;
  baseEditNames.push_back("ins");
  baseEditNames.push_back("del");
  if (_useMatch) {
    baseEditNames.push_back("mat");
    // If _identicalUnigramsOnly is true, we won't have a Sub11 operation, so
    // we'll only need a sub state if we are dealing with longer phrases.
    if (!_identicalUnigramsOnly || _maxSourcePhraseLength > 1 ||
        _maxTargetPhraseLength > 1) {
      baseEditNames.push_back("sub");
    }    
  }
  else
    baseEditNames.push_back("rep");
  
  _states.push_back(new StateType(_states.size(), start)); // start state

  // Create all state bigrams, except *->sta.
  BOOST_FOREACH(const string& dest, baseEditNames) {
    _states.push_back(new StateType(_states.size(), start + trans + dest));
    BOOST_FOREACH(const string& source, baseEditNames) {
      // Disallow ins->del if the --allow-redundant flag is absent.
      if (!_allowRedundant && source == "ins" && dest == "del")
        continue;
      _states.push_back(new StateType(_states.size(), source + trans + dest));
    }
  }
  
  string opBaseName = "Del";
  BOOST_FOREACH(const StateType& dest, _states) {
    // Nothing transitions to the single start state.
    if (dest.getName() == start)
      continue;
    for (int sourceLen = 1; sourceLen <= _maxSourcePhraseLength; ++sourceLen) {
      const string opName = opBaseName + lexical_cast<string>(sourceLen);
      if (iends_with(dest.getName(), opBaseName)) {
        EditOperation* op = new OpDelete(_ops.size(), &dest, opName, sourceLen);
        _ops.push_back(op);
        const string destFirst = dest.getName().substr(0, 3);
        BOOST_FOREACH(StateType& source, _states) {
          // If this is not a transition to start/finish and the last state name
          // in the source matches the first state name in the destination:
          if (!iends_with(source.getName(), transToStart) &&
              iends_with(source.getName(), destFirst)) {
            source.addValidOperation(op);
          }
        }
      }
    }
  }
  
  opBaseName = "Ins";
  BOOST_FOREACH(const StateType& dest, _states) {
    // Nothing transitions to the single start state.
    if (dest.getName() == start)
      continue;
    for (int targetLen = 1; targetLen <= _maxTargetPhraseLength; ++targetLen) {
      const string opName = opBaseName + lexical_cast<string>(targetLen);
      if (iends_with(dest.getName(), opBaseName)) {
        EditOperation* op = new OpInsert(_ops.size(), &dest, opName, targetLen);
        _ops.push_back(op);
        const string destFirst = dest.getName().substr(0, 3);
        BOOST_FOREACH(StateType& source, _states) {
          // If this is not a transition to start/finish and the last state name
          // in the source matches the first state name in the destination:
          if (!iends_with(source.getName(), transToStart) &&
              iends_with(source.getName(), destFirst)) {
            source.addValidOperation(op);
          }
        }
      }
    }
  }
    
  opBaseName = _useMatch ? "Sub" : "Rep";
  BOOST_FOREACH(const StateType& dest, _states) {
    // Nothing transitions to the single start state.
    if (dest.getName() == start)
      continue;
    for (int sourceLen = 1; sourceLen <= _maxSourcePhraseLength;
        ++sourceLen) {
      for (int targetLen = 1; targetLen <= _maxTargetPhraseLength;
          ++targetLen) {
        const string opName = opBaseName + lexical_cast<string>(sourceLen) +
            lexical_cast<string>(targetLen);
        if (iends_with(dest.getName(), opBaseName)) {
          EditOperation* op = 0;
          if (_useMatch) {
            // If _identicalUnigramsOnly is true, omit Sub operations for pairs
            // of unigrams.
            if (!_identicalUnigramsOnly || sourceLen > 1 || targetLen > 1) {
              op = new OpSubstitute(_ops.size(), &dest, opName, sourceLen,
                  targetLen);
            }
          }
          else {
            op = new OpReplace(_ops.size(), &dest, opName, sourceLen,
                targetLen);
          }
          if (op) {
            _ops.push_back(op);
            const string destFirst = dest.getName().substr(0, 3);
            BOOST_FOREACH(StateType& source, _states) {
              if (!iends_with(source.getName(), transToStart) &&
                  iends_with(source.getName(), destFirst)) {
                source.addValidOperation(op);
              }
            }
          }
        }
      }
    }
  }
  
  if (_useMatch) {
    opBaseName = "Mat";
    // We can't possibly match 1-2, 2-1, etc.
    const int maxMatchLength = std::min(_maxSourcePhraseLength,
        _maxTargetPhraseLength);
    BOOST_FOREACH(const StateType& dest, _states) {
      // Nothing transitions to the single start state.
      if (dest.getName() == start)
        continue;
      for (int len = 1; len <= maxMatchLength; ++len) {
        const string opName = opBaseName + lexical_cast<string>(len);
        if (iends_with(dest.getName(), opBaseName)) {
          EditOperation* op = new OpMatch(_ops.size(), &dest, opName, len);
          _ops.push_back(op);
          const string destFirst = dest.getName().substr(0, 3);
          BOOST_FOREACH(StateType& source, _states) {
            if (!iends_with(source.getName(), transToStart) &&
                iends_with(source.getName(), destFirst)) {
              source.addValidOperation(op);
            }
          }
        }
      }
    }
  }
}

template <typename Graph>
int StringEditModel<Graph>::processOptions(int argc, char** argv) {
  using namespace std;
  namespace opt = boost::program_options;
  std::string nglexSourceFname, nglexTargetFname;
  opt::options_description options(name() + " options");
  string fgenNameLat;
  options.add_options()
    ("allow-redundant", opt::bool_switch(&_allowRedundant),
        "if true, a delete operation may follow an insert operation \
(note: this is automatically true for a zero-order model)")
    ("cache-graphs", opt::bool_switch(&_cacheGraphs),
        "if true, store the graphs and rescore them instead of rebuilding")
    ("fv-cache-only", opt::bool_switch(&_cacheObsFeatsOnly),
        "if true, cache observed feature vectors, but not graphs")
    ("exact-match-state", opt::bool_switch(&_useMatch), "if true, use a \
match state when identical source and target phrases are encountered, or a \
substitute state if they differ; if false, use a replace state in both cases")
    ("identical-unigrams-only", opt::bool_switch(&_identicalUnigramsOnly),
        "exclude alignment features for unigrams that are not identical")
    ("no-final-arc-feats", opt::bool_switch(&_noFinalArcFeats),
        "if true, do not fire a feature for each arc in the FST that connects \
to the final state")
    ("order", opt::value<int>(&_order), "the Markov order")
    ("nglex-source", opt::value<string>(&nglexSourceFname),
        "file containing an n-gram lexicon for the source language")
    ("nglex-target", opt::value<string>(&nglexTargetFname),
        "file containing an n-gram lexicon for the target language")
    ("phrase-source", opt::value<int>(&_maxSourcePhraseLength),
        "maximum length of phrases on the source side")
    ("phrase-target", opt::value<int>(&_maxTargetPhraseLength),
        "maximum length of phrases on the target side")
    ("sample-cooc-n", opt::value<int>(&_featureCoocSamples)->default_value(0),
        "approximate feature co-occurrence counts by drawing n samples")
    ("start-arc", opt::bool_switch(&_includeStartArc),
        "if true, include a start arc in the FST")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help")) {
    cout << options << endl;
    return 0;
  }
  
  // If we're not extracting any features from the latent representation, then
  // there's no need to build a graph.
  if (_fgenAlign->isEmpty()) {
    assert(!_fgenObserved->isEmpty());
    // We need a dummy start state in order to keep AlignmentHypergraph happy.
    _states.push_back(new StateType(0, "sta"));
    return 0;
  }

  if (_order < 0 || _order > 2) {
    cout << "Invalid arguments: --order can be 0, 1, or 2 in this version\n";
    return 1;
  }
  
  if (_order == 0)
    addZeroOrderStates();
  else if (_order == 1)
    addFirstOrderStates();
  else {
    assert(_order == 2);
    addSecondOrderStates();
  }

  boost::shared_ptr<NgramLexicon> nglexSource, nglexTarget;
  bool lexiconInUse = false;
  if (nglexSourceFname.size() > 0) {
    nglexSource.reset(new NgramLexicon(nglexSourceFname));
    lexiconInUse = true;
  }
  if (nglexTargetFname.size() > 0) {
    nglexTarget.reset(new NgramLexicon(nglexTargetFname));
    lexiconInUse = true;
  }
  if (lexiconInUse) {
    BOOST_FOREACH(EditOperation& op, _ops)
      op.setNgramLexicons(nglexSource, nglexTarget);
  }

#if 0
  // Print information about the edit operations and states that were added.
  BOOST_FOREACH(const StateType& state, _states) {
    cout << state.getName() << ":\n";
    BOOST_FOREACH(const EditOperation* op, state.getValidOperations()) {
      cout << "\top:" << op->getName() << " \tdestState:" <<
        op->getDefaultDestinationState()->getName() << endl;
    }
  }
#endif
  
  // This ensures that Model::getCacheEnabled() returns true, even if only the
  // observed feature vectors are being cached. 
  if (_cacheObsFeatsOnly)
    _cacheGraphs = true;
  
  return 0;
}

template <typename Graph>
size_t StringEditModel<Graph>::gatherFeatures(const Pattern& x,
    const Label y) {
  WeightVector wNull;
  Graph* graph = getGraph(_fstCache, wNull, x, y);
  const int numEdges = graph->numEdges();
  return numEdges;
}

template <typename Graph>
LogWeight StringEditModel<Graph>::totalMass(const WeightVector& w,
    const Pattern& x, const Label y) {
  Graph* graph = getGraph(_fstCache, w, x, y);
  const LogWeight logZ = Inference<LogSemiring>::logPartition(*graph);
  return logZ;
}

template <typename Graph>
double StringEditModel<Graph>::viterbiScore(const WeightVector& w,
    const Pattern& x, const Label y) {
  Graph* graph = getGraph(_fstCache, w, x, y);
  const double maxScore = Inference<ViterbiSemiring>::viterbiScore(*graph);
  return maxScore;
}

template <typename Graph>
double StringEditModel<Graph>::maxFeatures(const WeightVector& w,
    SparseRealVec* fv, const Pattern& x, const Label y, bool includeObsFeats) {
  Graph* graph = 0;
  if (includeObsFeats)
    graph = getGraph(_fstCache, w, x, y, includeObsFeats);
  else
    graph = getGraph(_fstCacheNoObs, w, x, y, includeObsFeats);
  const double maxScore = Inference<ViterbiSemiring>::maxFeatureVector(
      *graph, fv);
  return maxScore;
}

template <typename Graph>
LogWeight StringEditModel<Graph>::expectedFeatures(const WeightVector& w,
    SparseLogVec* fv, const Pattern& x, const Label y, bool normalize) {
  Graph* graph = getGraph(_fstCache, w, x, y);
  const LogWeight logZ = Inference<LogSemiring>::logExpectedFeatures(
      *graph, fv);
  if (normalize)
    *fv /= logZ;  
  return logZ;
}

template <typename Graph>
LogWeight StringEditModel<Graph>::expectedFeatureCooccurrences(
    const WeightVector& w, AccumLogMat* fm, SparseLogVec* fv,
    const Pattern& x, const Label y, bool normalize) {
  Graph* graph = getGraph(_fstCache, w, x, y);
    
  ExpectationSemiring::InsideOutsideResult result;
  result.rBar = fv;
  result.tBar = fm;
  if (_featureCoocSamples <= 0) {
    Inference<ExpectationSemiring>::logExpectedFeatureCooccurrences(*graph,
        result);
  }
  else {
    Inference<ExpectationSemiring>::logExpectedFeatureCooccurrencesSample(
        *graph, _featureCoocSamples, result);
  }

  if (normalize) {
    *fv /= result.Z;
    *fm /= result.Z;
  }
  return result.Z;
}

template <typename Graph>
void StringEditModel<Graph>::alignmentsToMaxFvs(
    const std::vector<std::list<const Hyperedge*> >& alignments,
    boost::shared_ptr<std::vector<boost::shared_ptr<SparseRealVec> > >& maxFvs,
    int dim) {
  using namespace std;
  maxFvs.reset(new vector<boost::shared_ptr<SparseRealVec> >());
  BOOST_FOREACH(const list<const Hyperedge*>& path, alignments) {
    boost::shared_ptr<SparseRealVec> fv(new SparseRealVec(dim));
    BOOST_FOREACH(const Hyperedge* e, path) {
      assert(e->getChildren().size() == 1); // Only works for graphs at this point
      ublas_util::addExponentiated(*e->getFeatureVector(), *fv);
    }
    maxFvs->push_back(fv);
  }
}

template <typename Graph>
void StringEditModel<Graph>::getBestAlignments(std::ostream& str,
    boost::shared_ptr<std::vector<boost::shared_ptr<SparseRealVec> > >& maxFvs,
    const WeightVector& w, const Pattern& x, const Label y, bool includeObs) {
  using namespace std;
  
  Graph* graph = getGraph(_fstCache, w, x, y, includeObs);
  assert(graph);
  
  vector<list<const Hyperedge*> > alignments;
  
  if (KBestViterbiSemiring::k <= 0) {
    list<const Hyperedge*> alignment;
    Inference<ViterbiSemiring>::viterbiPath(*graph, alignment);
    alignments.push_back(alignment);
  }
  else {
    Inference<KBestViterbiSemiring>::viterbiPathsK(*graph, alignments);
    assert(alignments.size() <= KBestViterbiSemiring::k);
  }
  alignmentsToMaxFvs(alignments, maxFvs, w.getDim());
  
  const StringPair& pair = (StringPair&)x;
  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();

  for (int k = 0; k < alignments.size(); k++) {
    const list<const Hyperedge*>& edges = alignments[k];
    LogWeight weight(1);
    BOOST_FOREACH(const Hyperedge* edge, edges)
      weight *= edge->getWeight();
    str << "Alignment " << k << " ( weight = " << weight << " ):" << endl;
    stringstream alignedSource, alignedTarget;
    int i = 0, j = 0, iNew = -1, jNew = -1;
    size_t alignPos = 0;
    const StateType* source = &_states.front();
    assert(source->getName() == "sta");
    
    BOOST_FOREACH(const Hyperedge* edge, edges) {
      const int opId = edge->getLabel();
      if (opId == OpNone::ID)
        continue;
      assert(opId >= 0);      
      
      // FIXME: This inner loop is inefficient. We should create a lookup table
      // that maps op ids to ops.
      BOOST_FOREACH(const EditOperation* op, source->getValidOperations()) {
        if (op->getId() == opId) {
          source = op->apply(s, t, source, i, j, iNew, jNew);
          assert(source);
          assert(source->getId() >= 0);
          assert(iNew >= i && jNew >= j);
          int iPhraseLen = iNew - i;
          int jPhraseLen = jNew - j;
          
          str << op->getName() << " "; // Print the name of the edit operation.
          alignedSource << "|";
          alignedTarget << "|";
          
          if (iNew > i) {
            alignedSource << s[i];
            for (size_t k = i + 1; k < iNew; k++)
              alignedSource << " " << s[k];
          }
          if (jNew > j) {
            alignedTarget << t[j];
            for (size_t k = j + 1; k < jNew; k++)
              alignedTarget << " " << t[k];
          }
          
          if (jPhraseLen < iPhraseLen) {
            alignedTarget << (jPhraseLen > 0 ? "  " : " ");
            for (size_t k = jPhraseLen + 1; k < iPhraseLen; k++)
              alignedTarget << " " << " ";
          }
          else if (iPhraseLen < jPhraseLen) {
            alignedSource << (iPhraseLen > 0 ? "  " : " ");
            for (size_t k = iPhraseLen + 1; k < jPhraseLen; k++)
              alignedSource << " " << " ";
          }
  
          i = iNew;
          j = jNew;
          alignPos++;
          break;
        }
      }
    }
    str << endl;
    
    // Print the strings with alignment markers.
    str << alignedSource.str() << endl;
    str << alignedTarget.str() << endl;
  }
}

template <typename Graph>
void StringEditModel<Graph>::emptyCache() {
  _fstCache.clear();
  _fstCacheNoObs.clear();
  _fvCacheObs.clear();
}

template <typename Graph>
Graph* StringEditModel<Graph>::getGraph(boost::ptr_map<ExampleId, Graph>& cache,
    const WeightVector& w, const Pattern& x, const Label y, bool includeObs) {
  assert(_states.size() > 0);
  Graph* graph = 0;
  if (_cacheGraphs && !_cacheObsFeatsOnly &&
      x.getId() >= _onlyCacheIdsGreaterThanOrEqualTo) {
    ExampleId id = std::make_pair(x.getHashString(), y);
    typename boost::ptr_map<ExampleId, Graph>::iterator it = cache.find(id);
    if (it == cache.end()) {
      graph = new Graph(_states, _fgenAlign, _fgenObserved, !_noFinalArcFeats);
      assert(graph);
      graph->build(w, x, y, _includeStartArc, includeObs);
      graph->clearBuildVariables();
      cache.insert(id, graph);
    }
    else {
      graph = it->second;
      assert(graph);
      graph->rescore(w);
    }
  }
  else {
    if (!_reusableGraph) {
      // Initialize a "reusable" graph.
      _reusableGraph.reset(new Graph(_states, _fgenAlign, _fgenObserved,
          !_noFinalArcFeats));
    }
    assert(_reusableGraph);
    _reusableGraph->build(w, x, y, _includeStartArc, includeObs);
    graph = _reusableGraph.get();
  }
  assert(graph);
  return graph;
}

template <typename Graph>
boost::shared_ptr<const SparseRealVec> StringEditModel<Graph>::observedFeatures(
    const Pattern& x, const Label y) {
  using namespace boost;
  shared_ptr<const SparseRealVec> fv;
  if ((_cacheGraphs || _cacheObsFeatsOnly) &&
      x.getId() >= _onlyCacheIdsGreaterThanOrEqualTo) {
    ExampleId id = std::make_pair(x.getHashString(), y);
    unordered_map<ExampleId, shared_ptr<const SparseRealVec> >::iterator it =
        _fvCacheObs.find(id);
    if (it == _fvCacheObs.end()) {
      const SparseRealVec* fvRaw = _fgenObserved->getFeatures(x, y);
      assert(fvRaw);
      fv.reset(fvRaw);
      _fvCacheObs.insert(std::make_pair(id, fv));
    }
    else {
      fv = it->second;
      assert(fv);
    }
  }
  else {
    fv.reset(_fgenObserved->getFeatures(x, y));
  }
  assert(fv);
  return fv;
}

#endif
