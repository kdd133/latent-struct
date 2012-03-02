/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentFeatureGen.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "BmrmOptimizer.h"
#include "CognatePairReader.h"
#include "Dataset.h"
#include "EmOptimizer.h"
#include "EmptyAlignmentFeatureGen.h"
#include "EmptyObservedFeatureGen.h"
#include "Example.h"
#include "FeatureVectorPool.h"
#include "InputReader.h"
#include "KlementievRothWordFeatureGen.h"
#include "KlementievRothSentenceFeatureGen.h"
#include "Label.h"
#include "LbfgsOptimizer.h"
#include "LogFeatArc.h"
#include "LogLinearBinary.h"
#include "LogLinearBinaryObs.h"
#include "LogLinearMulti.h"
#include "MaxMarginBinary.h"
#include "MaxMarginBinaryObs.h"
#include "MaxMarginMulti.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Optimizer.h"
#include "Pattern.h"
#include "SentenceAlignmentFeatureGen.h"
#include "SentencePairReader.h"
#include "WordAlignmentFeatureGen.h"
#include "StringEditModel.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "WeightVector.h"
#include "WordPairReader.h"
#include <algorithm>
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/nullable.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
using namespace boost;
using namespace std;

int main(int argc, char** argv) {
  // Store the options in string format for later writing to an output file.
  stringstream optsStream;
  for (int i = 1; i < argc; i++)
    optsStream << argv[i] << " ";
  optsStream << endl;
  
  cout << optsStream.str(); // Print the options to stdout.

  // Parse the options.
  namespace opt = boost::program_options;
  const string blank("<NONE>");
  string dirPath("./");
  string evalFilename(blank);
  string fgenNameLat(blank);
  string fgenNameObs(blank);
  string loadDir(blank);
  string modelName(blank);
  string objName(blank);
  string optName(blank);
  string readerName(blank);
  string trainFilename(blank);
  string weightsInit(blank);
  int seed = 0;
  size_t poolMB = 0;
  size_t threads = 1;
  double trainFraction = 1.0;
  bool optEM = false;
  bool disablePool = false;
  bool split = false;
  bool sampleAllPositives = false;
  const string optAuto = "Auto";
  
  // Enumerate the choices for each option that involves a class name.
  const string CMA = ", ";
  stringstream fgenMsgLat;
  fgenMsgLat << "feature gen latent {" << EmptyAlignmentFeatureGen::name() <<
      CMA << SentenceAlignmentFeatureGen::name() <<
      CMA << WordAlignmentFeatureGen::name() << "}";  
  stringstream fgenMsgObs;
  fgenMsgObs << "feature gen observed {" << BiasFeatureGen::name() << CMA
      << EmptyObservedFeatureGen::name() << CMA
      << KlementievRothWordFeatureGen::name() << CMA
      << KlementievRothSentenceFeatureGen::name() << "}";      
  stringstream modelMsgObs;
  modelMsgObs << "model {" << StringEditModel<LogFeatArc>::name() << "}";
  stringstream objMsgObs;
  objMsgObs << "obj {" << LogLinearBinary::name() << CMA <<
      LogLinearBinaryObs::name() << CMA << LogLinearMulti::name() << CMA <<
      MaxMarginBinary::name() << CMA <<
      MaxMarginBinaryObs::name() << CMA << MaxMarginMulti::name() << "}";
  stringstream optMsgObs;
  optMsgObs << "opt {" << optAuto << CMA << BmrmOptimizer::name() << CMA <<
      LbfgsOptimizer::name() << "}";
  stringstream readerMsg;
  readerMsg << "reader that parses lines from input file {" <<
      CognatePairReader::name() << CMA << SentencePairReader::name() << CMA <<
      WordPairReader::name() << "}";  
  
  opt::options_description options("Main options");
  options.add_options()
    ("dir", opt::value<string>(&dirPath),
        "directory in which to store results (must already exist)")
    ("disable-pool", opt::bool_switch(&disablePool), "disable the memory pool")
//  ("em", opt::bool_switch(&optEM), "employ the optimizer inside an EM loop")
    ("eval", opt::value<string>(&evalFilename), "evaluation data file")
    ("fgen-lat", opt::value<string>(&fgenNameLat)->default_value(
        EmptyAlignmentFeatureGen::name()), fgenMsgLat.str().c_str())
    ("fgen-obs", opt::value<string>(&fgenNameObs)->default_value(
        EmptyObservedFeatureGen::name()), fgenMsgObs.str().c_str())
    ("load-dir", opt::value<string>(&loadDir),
        "load weights and features from directory and predict on eval data")
    ("model", opt::value<string>(&modelName)->default_value(
        StringEditModel<LogFeatArc>::name()), modelMsgObs.str().c_str())
    ("obj", opt::value<string>(&objName)->default_value(
        LogLinearMulti::name()), objMsgObs.str().c_str())
    ("opt", opt::value<string>(&optName)->default_value(optAuto),
        optMsgObs.str().c_str())
    ("pool-size", opt::value<size_t>(&poolMB)->default_value(25),
        "max size of pool, *per thread* (megabytes)")
    ("reader", opt::value<string>(&readerName), readerMsg.str().c_str())
    ("sample-all-positives", opt::bool_switch(&sampleAllPositives),
        "if --sample-train is enabled, this option ensures that all the \
positive examples present in the data are retained")  
    ("sample-train", opt::value<double>(&trainFraction),
        "learn on this fraction of the train data (uniformly sampled, \
without replacement); if greater than 1, the value is interpreted as the \
*number* of training examples to sample")
    ("seed", opt::value<int>(&seed)->default_value(0),
        "seed for random number generator")
    ("split", opt::bool_switch(&split), "used in combination with \
sample-train to use the unselected training examples as the eval set")
    ("threads", opt::value<size_t>(&threads)->default_value(1),
        "number of threads employed to parallelize computations")
    ("train", opt::value<string>(&trainFilename), "training data file")
    ("weights-init", opt::value<string>(&weightsInit)->default_value("noise"),
        "initialize weights {heuristic, heuristic+noise, noise, zero}")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  const bool help = vm.count("help");
  const bool load = vm.count("load-dir");
  const bool writeFiles = vm.count("dir");
  
  if (split && trainFraction == 1.0) {
    cout << "Invalid arguments: Can't have --split with --sample-train=1.0\n"
        << options << endl;
    return 1;
  }
  if (load && !vm.count("eval")) {
    cout << "Invalid arguments: Can't have --load-dir without --eval\n"
        << options << endl;
    return 1;
  }
  if (load && !vm.count("dir")) {
    cout << "Invalid arguments: Can't have --load-dir without --dir\n"
        << options << endl;
    return 1;
  }
  
  if (!boost::iends_with(dirPath, "/"))
    dirPath += "/";
  if (load && !boost::iends_with(loadDir, "/"))
    loadDir += "/";
    
  if (load) {
    if (dirPath == loadDir) {
      cout << "Invalid arguments: --load-dir and --dir cannot give same paths\n"
          << options << endl;
      return 1;
    }
  }
  
  bool cachingEnabled = false;
  vector<Model*> models;
  shared_ptr<Alphabet> loadedAlphabet(new Alphabet(false, false));
  if (load) {
    if (!loadedAlphabet->read(loadDir + "alphabet.txt")) {
      cout << "Error: Unable to read " << loadDir << "alphabet.txt" << endl;
      return 1;
    }
    loadedAlphabet->lock();
  }
  for (size_t th = 0; th < threads; th++) {
    shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
    if (load)
      alphabet = loadedAlphabet;
    else
      alphabet->unlock(); // in preparation for feature gathering
      
    // initialize the latent feature generator
    shared_ptr<AlignmentFeatureGen> fgenLat;
    if (fgenNameLat == WordAlignmentFeatureGen::name())
      fgenLat.reset(new WordAlignmentFeatureGen(alphabet));
    else if (fgenNameLat == SentenceAlignmentFeatureGen::name())
      fgenLat.reset(new SentenceAlignmentFeatureGen(alphabet));
    else if (fgenNameLat == EmptyAlignmentFeatureGen::name())
      fgenLat.reset(new EmptyAlignmentFeatureGen(alphabet));
    else {
      if (!help) {
        cout << "Invalid arguments: An unrecognized fgen-lat name was given: "
            << fgenNameLat << endl << options << endl;
        return 1;
      }
    }
    if (fgenLat != 0 && fgenLat->processOptions(argc, argv)) {
      cout << "AlignmentFeatureGen::processOptions() failed." << endl;
      return 1;
    }
    
    // initialize the observed feature generator
    shared_ptr<ObservedFeatureGen> fgenObs;
    if (fgenNameObs == BiasFeatureGen::name())
      fgenObs.reset(new BiasFeatureGen(alphabet));
    else if (fgenNameObs == KlementievRothWordFeatureGen::name())
      fgenObs.reset(new KlementievRothWordFeatureGen(alphabet));
    else if (fgenNameObs == KlementievRothSentenceFeatureGen::name())
      fgenObs.reset(new KlementievRothSentenceFeatureGen(alphabet));
    else if (fgenNameObs == EmptyObservedFeatureGen::name())
      fgenObs.reset(new EmptyObservedFeatureGen(alphabet));
    else {
      if (!help) {
        cout << "Invalid arguments: An unrecognized fgen-obs name was given: "
            << fgenNameObs << endl << options << endl;
        return 1;
      }
    }
    if (fgenObs != 0 && fgenObs->processOptions(argc, argv)) {
      cout << "ObservedFeatureGen::processOptions() failed." << endl;
      return 1;
    }
    
    // initialize a model
    Model* model = 0;
    if (modelName == StringEditModel<StdFeatArc>::name()) {
      if (istarts_with(objName, "LogLinear"))
        model = new StringEditModel<LogFeatArc>(fgenLat, fgenObs);
      else if (istarts_with(objName, "MaxMargin"))
        model = new StringEditModel<StdFeatArc>(fgenLat, fgenObs);
      else if (!help) {
        cout << "Invalid arguments: Objective name does not begin with LogLinear "
            << "or MaxMargin: " << objName << endl << options << endl;
        return 1;
      }
    }
    else {
      if (!help) {
        cout << "Invalid arguments: An unrecognized model name was given: "
            << modelName << endl << options << endl;
        return 1;
      }
    }
    if (model != 0 && model->processOptions(argc, argv)) {
      cout << "Model::processOptions() failed." << endl;
      delete model;
      return 1;
    }
    
    if (model != 0) {
      if (!cachingEnabled)
        cachingEnabled = model->getCacheEnabled();
      model->setCacheEnabled(false); // Disable caching during feature gathering    
      models.push_back(model);
    }
  }
  
  boost::timer::auto_cpu_timer timerTotal;
  
  Dataset trainData(threads);
  Dataset evalData(threads);
  
  // initialize the input reader
  scoped_ptr<InputReader> reader;
  if (readerName == CognatePairReader::name())
    reader.reset(new CognatePairReader());
  else if (readerName == SentencePairReader::name())
    reader.reset(new SentencePairReader());
  else if (readerName == WordPairReader::name())
    reader.reset(new WordPairReader()); 
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized reader name was given: "
          << readerName << endl << options << endl;
      return 1;
    }
  }
  
  if (!help && vm.count("train")) {
    cout << "Loading " << trainFilename << " ...\n";
    boost::timer::auto_cpu_timer loadTrainTimer;
    if (Utility::loadDataset(*reader, trainFilename, trainData)) {
      cout << "Error: Unable to load train file " << trainFilename << endl;
      return 1;
    }
    cout << "Read " << trainData.numExamples() << " train examples, " <<
        trainData.getLabelSet().size() << " classes\n";
  
    if (trainFraction < 1.0 || (trainFraction > 1.0 &&
        (size_t)trainFraction < trainData.numExamples())) {
      size_t subTrainSize;
      if (trainFraction < 1)
        subTrainSize = (size_t)(trainFraction * trainData.numExamples());
      else
        subTrainSize = (size_t)trainFraction; // interpret trainFraction as num
      cout << "Selecting a random sample of " << subTrainSize <<
          " train examples.\n";
      mt19937 mt(seed);
      uniform_int<> uni(0, trainData.numExamples()-1);
      variate_generator<mt19937, uniform_int<> > rgen(mt, uni);
      set<int> selected;
      if (sampleAllPositives) {
        for (size_t i = 0; i < trainData.numExamples(); i++) {
          if (trainData.getExamples()[i].y() == TrainingObjective::kPositive)
            selected.insert((int)i);
        }
      }
      if (selected.size() >= subTrainSize) {
        cout << "Invalid arguments: There are more positive examples than " <<
            "the total number of training examples you asked to sample. " <<
            "Please lower the value of --sample-train or remove the " <<
            "--sample-all-positives flag." << endl;
        return 1;
      }
      while (selected.size() < subTrainSize)
        selected.insert(rgen());
      Dataset subTrainData(threads);
      if (split) {
        // Put selected examples in train, unselected in eval.
        for (size_t i = 0; i < trainData.numExamples(); i++) {
          if (selected.find(i) == selected.end())
            evalData.addExample(trainData.getExamples()[i]);
          else
            subTrainData.addExample(trainData.getExamples()[i]);
        }
        assert(evalData.numExamples() == trainData.numExamples()-subTrainSize);
      }
      else {
        BOOST_FOREACH(int i, selected) {
          subTrainData.addExample(trainData.getExamples()[i]);
        }
      }
      trainData = subTrainData;
    }
  
    if ((int)trainData.getLabelSet().size() <= 1) {
      cout << "Error: Fewer than 2 class labels were found.\n";
      return 1;
    }
    BOOST_FOREACH(const Label y, trainData.getLabelSet()) {
      if (y >= (int)trainData.getLabelSet().size()) {
        cout << "Error: The class labels are not sequential (0,1,...,k).\n";
        return 1;
      }
    }
  }
  
  // Initialize the training objective.
  shared_ptr<TrainingObjective> objective;
  if (objName == LogLinearMulti::name())
    objective.reset(new LogLinearMulti(trainData, models));
  else if (objName == LogLinearBinary::name())
    objective.reset(new LogLinearBinary(trainData, models));
  else if (objName == LogLinearBinaryObs::name())
    objective.reset(new LogLinearBinaryObs(trainData, models));
  else if (objName == MaxMarginBinary::name()) {
    objective.reset(new MaxMarginBinary(trainData, models));
    optEM = true; // EM is currently required for optimizing this objective
  }
  else if (objName == MaxMarginBinaryObs::name())
    objective.reset(new MaxMarginBinaryObs(trainData, models));
  else if (objName == MaxMarginMulti::name()) {
    objective.reset(new MaxMarginMulti(trainData, models));
    optEM = true; // EM is currently required for optimizing this objective
  }
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized objective name was given: "
          << objName << endl << options << endl;
      return 1;
    }
  }
  // Note: If --help is enabled, the data will not have been loaded
  assert(help || threads == objective->getNumModels());

  if (!help && load) {
    assert(writeFiles);
    assert(loadedAlphabet->size() > 0);
    WeightVector w;
    if (!w.read(loadDir + "weights.txt", loadedAlphabet->size())) {
      cout << "Error: Unable to read " << loadDir << "weights.txt" << endl;
      return 1;
    }
    assert(w.getDim() == (int)loadedAlphabet->size());
    {
      cout << "Loading " << evalFilename << " ...\n";
      boost::timer::auto_cpu_timer loadEvalTimer;
      if (Utility::loadDataset(*reader, evalFilename, evalData)) {
        cout << "Error: Unable to load eval file " << evalFilename << endl;
        return 1;
      }
      cout << "Read " << evalData.numExamples() << " eval examples, " <<
          evalData.getLabelSet().size() << " classes\n";
    }
    {
      cout << "Classifying eval examples ...\n";
      boost::timer::auto_cpu_timer classifyEvalTimer;
      string fname = dirPath + "eval_predictions.txt";
      Utility::evaluate(w, *objective, evalData, "Eval", fname);
    }
    // Write the command line options to a file.
    const string optsFname = dirPath + "options.txt";
    ofstream optsOut(optsFname.c_str());
    if (optsOut.good()) {
      optsOut << optsStream.str();
      optsOut.close();
    }
    else
      cout << "Warning: Unable to write " << optsFname << endl;
    return 0;
  }

  // Initialize the optimizer.
  shared_ptr<Optimizer> optimizer;
  if (optName == optAuto) {
    // Automatically select an appropriate optimizer for the chosen objective.
    if (istarts_with(objName, "LogLinear"))
      optimizer.reset(new LbfgsOptimizer(*objective));
    else if (istarts_with(objName, "MaxMargin"))
      optimizer.reset(new BmrmOptimizer(*objective));
    else {
      cout << "Automatic optimizer selection failed for " << objName << endl;
      return 1;
    }
  }
  else if (optName == LbfgsOptimizer::name())    
    optimizer.reset(new LbfgsOptimizer(*objective));
  else if (optName == BmrmOptimizer::name())
    optimizer.reset(new BmrmOptimizer(*objective));
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized optimizer name was given: "
          << optName << endl << options << endl;
      return 1;
    }
  }
  if (optimizer->processOptions(argc, argv)) {
    cout << "Optimizer::processOptions() failed." << endl;
    return 1;
  }

  // Wrap the optimizer in an EM procedure if requested.
  shared_ptr<Optimizer> outerOpt;
  if (optEM) {
    outerOpt.reset(new EmOptimizer(*objective, optimizer));
    if (outerOpt->processOptions(argc, argv)) {
      cout << "outerOpt->processOptions() failed." << endl;
      return 1;
    }
  }
  else
    outerOpt = optimizer; // processOptions already called for inner optimizer

  if (help) {
    cout << options << endl;
    return 1;
  }
  
  cout << "Gathering features ...\n";
  size_t maxNumFvs = 0, totalNumFvs = 0;
  {
    boost::timer::auto_cpu_timer gatherTimer;
    objective->gatherFeatures(maxNumFvs, totalNumFvs);
    assert(maxNumFvs > 0 && totalNumFvs > 0);
    objective->combineAndLockAlphabets();
  }
  
  shared_ptr<const AlignmentFeatureGen> fgenLat =
      objective->getModel(0).getFgenLatent();
  shared_ptr<const ObservedFeatureGen> fgenObs =
      objective->getModel(0).getFgenObserved();
  shared_ptr<Alphabet> alphabet = fgenLat->getAlphabet();
  assert(alphabet == fgenObs->getAlphabet()); // Assume the alphabet is shared
  
  const int d = alphabet->size();
  if (d == 0) {
    cout << "Error: No features were found!\n";
    return 1;
  }
  cout << "Extracted " << d << " features\n";  
  WeightVector w(d);

  // Enable caching at this point, if requested.
  if (cachingEnabled)
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      objective->getModel(i).setCacheEnabled(true);
      // The cache may contain a "reusable" fst: see StringEditModel::getFst().
      objective->getModel(i).emptyCache();      
    }
  
  // Set initial weights (Note: the reAlloc above set them all to zero).
  if (weightsInit != "zero") {
    double* v = new double[d];
    if (istarts_with(weightsInit, "heuristic")) {
      const Alphabet::DictType& dict = alphabet->getDict();
      Alphabet::DictType::const_iterator it; 
      for (it = dict.begin(); it != dict.end(); it++)
        v[it->second] = fgenLat->getDefaultFeatureWeight(it->first);
    }
    if (iends_with(weightsInit, "noise")) {
      mt19937 mt(seed);
      normal_distribution<> gaussian(0, 0.01);
      variate_generator<mt19937, normal_distribution<> > rgen(mt, gaussian);
      for (int i = 0; i < d; i++)
        v[i] = rgen();
    }
    w.setWeights(v, d);
    delete[] v;
  }

  // Optionally enable memory pools for managing FeatureVector objects.
  ptr_vector<nullable<FeatureVectorPool<RealWeight> > > poolsLat;
  ptr_vector<nullable<FeatureVectorPool<RealWeight> > > poolsObs;
  if (!disablePool) {
    assert(objective->getNumModels() == threads);
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      Model& model = objective->getModel(i);
      const size_t maxLatEntries = model.getFgenLatent()->getMaxEntries();
      const size_t maxObsEntries = model.getFgenObserved()->getMaxEntries();    
      cout << "thread " << i << " maxLatEntries: " << maxLatEntries <<
          "  maxObsEntries: " << maxObsEntries << endl;
      const int bytesPerEntry = sizeof(int) + sizeof(double); // indices + values
      assert(poolMB > 0);
      const unsigned long long maxBytes = poolMB * 1048576ULL;
      assert(maxLatEntries + maxObsEntries > 0);
      
      if (maxLatEntries > 0) {
        const size_t maxAllowedLatFvs =
            maxBytes / (maxLatEntries * bytesPerEntry);
        cout << "Allocating pool of size " << min(maxAllowedLatFvs, maxNumFvs)
            << " for latent feature vectors.\n";
        // It doesn't make sense to combine the memory pool with caching of
        // transducers, since the feature vectors pointed to by the cached
        // transducers would be in constant flux. We can still use the pool when
        // we classify examples, though.
        poolsLat.push_back(new FeatureVectorPool<RealWeight>(maxLatEntries,
          min(maxAllowedLatFvs, maxNumFvs), false)); // not expandable
        if (!model.getCacheEnabled())
          model.getFgenLatent()->enablePool(&poolsLat[i]);
      }
      else
        poolsLat.push_back(0);
        
      if (maxObsEntries > 0) {
        // Generally (always?), only one observed feature vector appears in the
        // fst for each class, so we'll initialize this pool to size 1 and allow
        // it to expand. (1 per thread, that is)
        cout << "Allocating pool of size 1 for observed feature vectors.\n";
        // See comment in previous block.
        poolsObs.push_back(new FeatureVectorPool<RealWeight>(maxObsEntries, 1,
            true)); // expandable
        if (!model.getCacheEnabled())
          model.getFgenObserved()->enablePool(&poolsObs[i]);
      }
      else
        poolsObs.push_back(0);
    }
  }
  assert(disablePool || poolsLat.size() == threads);
  assert(disablePool || poolsObs.size() == threads);

  // Train the model.
  cout << "Calling Optimizer.train()\n";
  outerOpt->train(w);
  
  // If an output directory was specfied, write some files.
  if (writeFiles) {
    if (!alphabet->write(dirPath + "alphabet.txt"))
      cout << "Warning: Unable to write " << dirPath << "alphabet.txt" << endl;  
    if (!w.write(dirPath + "weights.txt"))
      cout << "Warning: Unable to write " << dirPath << "weights.txt" << endl;
    const string optsFname = dirPath + "options.txt";
    ofstream optsOut(optsFname.c_str());
    if (optsOut.good()) {
      optsOut << optsStream.str();
      optsOut.close();
    }
    else
      cout << "Warning: Unable to write " << optsFname << endl;
  }
  
  // Classify train examples and optionally write the predictions to a file.
  // Note: The model's transducer cache can still be used at this point, so we
  // defer purging it until after classifying the train examples.
  cout << "Classifying train examples ...\n";
  {
    boost::timer::auto_cpu_timer classifyTrainTimer;
    string fname = writeFiles ? dirPath + "train_predictions.txt" : "";
    Utility::evaluate(w, *objective, trainData, "Train", fname);
  }
  
  trainData.clear(); // We no longer need the training data.
  
  // If the model was caching transducers, we did not use the feature vector
  // pool during training. We'll enable it now (or simply reset to null if it
  // was never initialized) for the sake of classifying the eval examples.
  // TODO: This is actually inefficient in terms of memory use, since the pool
  // was allocated before training, but not used during training. It should
  // instead be allocated here if caching was also enabled.
  if (cachingEnabled) {
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      Model& model = objective->getModel(i);
      // No point in caching during evaluation.
      model.setCacheEnabled(false);
      model.emptyCache();
      if (!disablePool) {
        if (!poolsLat.is_null(i))
          model.getFgenLatent()->enablePool(&poolsLat[i]);
        if (!poolsObs.is_null(i))
          model.getFgenObserved()->enablePool(&poolsObs[i]);
      }
    }
  }
  
  // Classify eval examples and optionally write the predictions to a file.
  if (!split && vm.count("eval")) {
    cout << "Loading " << evalFilename << " ...\n";
    boost::timer::auto_cpu_timer loadEvalTimer;
    if (Utility::loadDataset(*reader, evalFilename, evalData)) {
      cout << "Error: Unable to load eval file " << evalFilename << endl;
      return 1;
    }
    cout << "Read " << evalData.numExamples() << " eval examples, " <<
        evalData.getLabelSet().size() << " classes\n";
  }
  assert(evalData.numExamples() > 0);
  cout << "Classifying eval examples ...\n";
  {
    boost::timer::auto_cpu_timer classifyEvalTimer;
    string fname = writeFiles ? dirPath + "eval_predictions.txt" : "";
    Utility::evaluate(w, *objective, evalData, "Eval", fname);
  }

  objective.reset(); // must be deleted before the memory pools
  return 0;
}
