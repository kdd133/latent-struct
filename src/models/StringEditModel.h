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
#include "AlignmentTransducer.h"
#include "FeatureVector.h"
#include "InputReader.h"
#include "Label.h"
#include "LogFeatArc.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "OpDelete.h"
#include "OpInsert.h"
#include "OpMatch.h"
#include "OpSubstitute.h"
#include "OpReplace.h"
#include "Pattern.h"
#include "StateType.h"
#include "StdFeatArc.h"
#include "StringEditModel.h"
#include "StringPair.h"
#include "Transducer.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/shared_ptr.hpp>
#include <list>
#include <set>
#include <string>
using namespace boost;
using namespace std;

template <typename Arc>
class StringEditModel : public Model {

  public:
  
    StringEditModel(shared_ptr<AlignmentFeatureGen> fgenAlign,
      shared_ptr<ObservedFeatureGen> fgenObserved);
    
    ~StringEditModel();
    
    virtual int processOptions(int argc, char** argv);
    
    virtual size_t gatherFeatures(const Pattern& pattern,
      const Label label);
    
    virtual LogWeight totalMass(const WeightVector& w, const Pattern& pattern,
      const Label label);
    
    virtual RealWeight viterbiScore(const WeightVector& w,
      const Pattern& pattern, const Label label);
  
    virtual RealWeight maxFeatures(const WeightVector& w,
      FeatureVector<RealWeight>& fv, const Pattern& pattern, const Label label,
      bool includeObservedFeaturesArc = true);
    
    virtual LogWeight expectedFeatures(const WeightVector& w,
      FeatureVector<LogWeight>& fv, const Pattern& pattern, const Label label,
      bool normalize = true);
      
    virtual FeatureVector<RealWeight>* observedFeatures(const Pattern& pattern,
      const Label label, bool& callerOwns);
    
    static const string& name() {
      static const string _name = "StringEdit";
      return _name;
    }
    
    virtual void emptyCache();
    
    typedef pair<size_t, Label> ExampleId;
    typedef AlignmentTransducer<Arc> Fst;


  private:
  
    // If true, the fst we build will consider exact matches, in addition to
    // generic substitutions.
    bool _useMatch;
    
    // If true, the fst we build will allow a deletion to follow an insertion.
    bool _allowRedundant;
    
    // Maximum phrase length on the source side.
    int _maxSourcePhraseLength;
    
    // Maximum phrase length on the target side.
    int _maxTargetPhraseLength;
    
    // A list of state types that comprise the nodes in an fst.
    list<StateType> _states;
    
    // A list of edit operations that comprise the arcs in an fst.
    list<const EditOperation*> _ops;
    
    // If true, do not fire any features for arc connecting to the final state.
    bool _noFinalArcFeats;

    // A cache for Fsts that include an arc for the observed features. 
    ptr_map<ExampleId, Fst> _fstCache;
    
    // A cache for Fsts that omit the arc for the observed features.
    ptr_map<ExampleId, Fst> _fstCacheNoObs;
    
    // A cache for observed feature vectors.
    ptr_map<ExampleId, FeatureVector<RealWeight> > _fvCacheObs;
    
    Fst* getFst(ptr_map<ExampleId, Fst>& cache, const WeightVector& w,
        const Pattern& x, const Label y,
        bool includeObsFeaturesArc = true);
    
    StringEditModel(const StringEditModel& x);
    StringEditModel& operator=(const StringEditModel& x);
};

template <typename Arc>
StringEditModel<Arc>::StringEditModel(shared_ptr<AlignmentFeatureGen> fgenAlign,
    shared_ptr<ObservedFeatureGen> fgenObserved) :
    Model(fgenAlign, fgenObserved), _useMatch(false), _allowRedundant(false),
    _maxSourcePhraseLength(1), _maxTargetPhraseLength(1),
    _noFinalArcFeats(false) {
}

template <typename Arc>
int StringEditModel<Arc>::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("allow-redundant", opt::bool_switch(&_allowRedundant),
        "if true, a delete operation may follow an insert operation")
    ("cache-fsts", opt::bool_switch(&_cacheFsts),
        "if true, store the FSTs and rescore them instead of rebuilding")
    ("exact-match-state", opt::bool_switch(&_useMatch), "if true, include an \
exact match state; otherwise, use a generic substitute only")
    ("no-final-arc-feats", opt::bool_switch(&_noFinalArcFeats),
        "if true, do not fire a feature for each arc in the FST that connects \
to the final state")
    ("phrase-source", opt::value<int>(&_maxSourcePhraseLength),
        "maximum length of phrases on the source side")
    ("phrase-target", opt::value<int>(&_maxTargetPhraseLength),
        "maximum length of phrases on the target side")
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
  
  StateType start(0, "start");
  StateType ins(1, "ins");
  StateType del(2, "del");
  StateType sub(3, "sub");
  StateType idn(4, "idn");
  
  _states.push_back(start);
  _states.push_back(ins);
  _states.push_back(del);
  if (_useMatch)
    _states.push_back(idn);
  _states.push_back(sub);
  
  // note: -1 means there is no restriction on what this op can follow
  int deleteNoFollow = -1;
  if (!_allowRedundant)
    deleteNoFollow = ins.getId();
  
  int opId = 0; // we will assign a unique identifier to each edit operation
  
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    _ops.push_back(new OpDelete(opId++, del.getId(), "Delete" +
        lexical_cast<string>(s), s, deleteNoFollow));
  }
  
  for (int t = 1; t <= _maxTargetPhraseLength; t++) {
    _ops.push_back(new OpInsert(opId++, ins.getId(), "Insert" +
        lexical_cast<string>(t), t));
  }
  
  for (int s = 1; s <= _maxSourcePhraseLength; s++) {
    for (int t = 1; t <= _maxTargetPhraseLength; t++) {
      if (_useMatch) {
        _ops.push_back(new OpSubstitute(opId++, sub.getId(), "Substitute" +
            lexical_cast<string>(s) + lexical_cast<string>(t), s, t));
        _ops.push_back(new OpMatch(opId++, idn.getId(), "Match" +
            lexical_cast<string>(s) + lexical_cast<string>(t), s, t));
      }
      else {
        _ops.push_back(new OpReplace(opId++, sub.getId(), "Replace" +
            lexical_cast<string>(s) + lexical_cast<string>(t), s, t));
      }
    }
  }
  
  return 0;
}

template <typename Arc>
StringEditModel<Arc>::~StringEditModel() {
  // dellocate the EditOperations that were created in processOptions()
  BOOST_FOREACH(const EditOperation* op, _ops) {
    assert(op != 0);
    delete op;
  }  
}

template <typename Arc>
size_t StringEditModel<Arc>::gatherFeatures(const Pattern& x,
    const Label y) {
  WeightVector wNull;
  Fst* fst = getFst(_fstCache, wNull, x, y);
  const int numArcs = fst->numArcs();
  fst->clearDynProgVariables();
  return numArcs;
}

template <typename Arc>
LogWeight StringEditModel<Arc>::totalMass(const WeightVector& w,
    const Pattern& x, const Label y) {
  Fst* fst = getFst(_fstCache, w, (StringPair&)x, y);
  const LogWeight logZ = fst->logPartition();
  fst->clearDynProgVariables();
  return logZ;
}

template <typename Arc>
RealWeight StringEditModel<Arc>::viterbiScore(const WeightVector& w,
    const Pattern& x, const Label y) {
  Fst* fst = getFst(_fstCache, w, x, y);
  FeatureVector<RealWeight> fv;
  const RealWeight maxScore = fst->maxFeatureVector(fv, true);
  fst->clearDynProgVariables();
  return maxScore;
}

template <typename Arc>
RealWeight StringEditModel<Arc>::maxFeatures(const WeightVector& w,
    FeatureVector<RealWeight>& fv, const Pattern& x, const Label y,
    bool includeObsFeats) {
  Fst* fst = 0;
  if (includeObsFeats)
    fst = getFst(_fstCache, w, (StringPair&)x, y, includeObsFeats);
  else
    fst = getFst(_fstCacheNoObs, w, (StringPair&)x, y, includeObsFeats);
  const RealWeight maxScore = fst->maxFeatureVector(fv);
  fst->clearDynProgVariables();
  return maxScore;
}

template <typename Arc>
LogWeight StringEditModel<Arc>::expectedFeatures(const WeightVector& w,
    FeatureVector<LogWeight>& fv, const Pattern& x, const Label y,
    bool normalize) {
  Fst* fst = getFst(_fstCache, w, (StringPair&)x, y);
  const LogWeight logZ = fst->logExpectedFeaturesUnnorm(fv);
  fst->clearDynProgVariables();
  if (normalize)
    fv.timesEquals(-logZ);
  return logZ;
}

template <typename Arc>
void StringEditModel<Arc>::emptyCache() {
  _fstCache.clear();
  _fstCacheNoObs.clear();
  _fvCacheObs.clear();
}

template <typename Arc>
AlignmentTransducer<Arc>* StringEditModel<Arc>::getFst(
    ptr_map<ExampleId, Fst>& cache, const WeightVector& w, const Pattern& x,
    const Label y, bool includeObs) {
  assert(_ops.size() > 0);
  assert(_states.size() > 0);
  Fst* fst = 0;
  if (_cacheFsts) {
    ExampleId id = make_pair(x.getId(), y);
    typename ptr_map<ExampleId, Fst>::iterator it = cache.find(id);
    if (it == cache.end()) {
      fst = new Fst(_states, _ops, _fgenAlign, _fgenObserved,
          !_noFinalArcFeats);
      assert(fst);
      fst->build(w, (const StringPair&)x, y, includeObs);
      fst->clearBuildVariables();
      cache.insert(id, fst);
    }
    else {
      fst = it->second;
      assert(fst);
      fst->rescore(w);
    }
  }
  else {
    if (cache.size() == 0) {
      // Initialize a "reusable" fst.
      ExampleId id = make_pair(0, 0);
      cache.insert(id, new Fst(_states, _ops, _fgenAlign, _fgenObserved,
          !_noFinalArcFeats));
    }
    assert(cache.size() == 1);
    fst = cache.begin()->second;
    assert(fst);
    fst->build(w, (const StringPair&)x, y, includeObs);
  }
  assert(fst);
  return fst;
}

template <typename Arc>
FeatureVector<RealWeight>* StringEditModel<Arc>::observedFeatures(
    const Pattern& x, const Label y, bool& callerOwns) {
  callerOwns = !_cacheFsts;
  FeatureVector<RealWeight>* fv = 0;
  if (_cacheFsts) {
    ExampleId id = make_pair(x.getId(), y);
    typename ptr_map<ExampleId, FeatureVector<RealWeight> >::iterator it =
        _fvCacheObs.find(id);
    if (it == _fvCacheObs.end()) {
      assert(_fgenObserved->getPool() == 0);
      fv = _fgenObserved->getFeatures(x, y);
      assert(fv);
      _fvCacheObs.insert(id, fv);
    }
    else {
      fv = it->second;
      assert(fv);
    }
  }
  else {
    fv = _fgenObserved->getFeatures(x, y);
  }
  assert(fv);
  return fv;
}

#endif
