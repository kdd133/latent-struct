/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentFeatureGen.h"
#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BergsmaKondrakWordFeatureGen.h"
#include "BiasFeatureGen.h"
#include "BmrmOptimizer.h"
#include "CognatePairReader.h"
#include "CognatePairAlignerReader.h"
#include "Dataset.h"
#include "EmOptimizer.h"
#include "EmptyAlignmentFeatureGen.h"
#include "EmptyObservedFeatureGen.h"
#include "Example.h"
#include "InputReader.h"
#include "KlementievRothSentenceFeatureGen.h"
#include "KlementievRothWordFeatureGen.h"
#include "Label.h"
#include "LbfgsOptimizer.h"
#include "LogLinearBinary.h"
#include "LogLinearBinaryObs.h"
#include "LogLinearBinaryUnscaled.h"
#include "LogLinearMulti.h"
#include "LogLinearMultiELFV.h"
#include "LogLinearMultiELFV_sigmoid.h"
#include "LogLinearMultiUW.h"
#include "MainHelpers.h"
#include "MaxMarginBinary.h"
#include "MaxMarginBinaryObs.h"
#include "MaxMarginMulti.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Optimizer.h"
#include "Parameters.h"
#include "Pattern.h"
#include "Regularizer.h"
#include "RegularizerL2.h"
#include "RegularizerNone.h"
#include "RegularizerSoftTying.h"
#include "SentenceAlignmentFeatureGen.h"
#include "SentencePairReader.h"
#include "StochasticGradientOptimizer.h"
#include "StringEditModel.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "ValidationSetHandler.h"
#include "WordAlignmentFeatureGen.h"
#include "WordPairReader.h"
#include <algorithm>
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
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
  bool addBeginEndMarkers = false;
  bool keepAllPositives = false;
  bool noEarlyGridStop = false;
  bool optEM = false;
  bool printAlignments = false;
  bool skipTraining = false;
  bool split = false;
  const string blank("<NONE>");
  const string optAuto("Auto");
  double negativeRatio = 0.0;
  double trainFraction = 1.0;
  double weightsNoise;
  int seed = 0;
  size_t threads = 1;
  string dirPath("./");
  string loadFeaturesFilename;
  string fgenNameLat(blank);
  string fgenNameObs(blank);
  string modelName(blank);
  string objName(blank);
  string optName(blank);
  string readerName(blank);
  string regName(blank);
  string saveFeaturesFilename;
  string trainFilename(blank);
  string validationFilename(blank);
  string weightsInit(blank);
  vector<double> betas;
  vector<double> tolerances;
  vector<string> evalFilenames;
  
  // Enumerate the choices for each option that involves a class name.
  const string CMA = ", ";
  stringstream fgenMsgLat;
  fgenMsgLat << "latent feature generator {"
      << EmptyAlignmentFeatureGen::name() <<
      CMA << SentenceAlignmentFeatureGen::name() <<
      CMA << WordAlignmentFeatureGen::name() << "}";  
  stringstream fgenMsgObs;
  fgenMsgObs << "observed feature generator {" << BiasFeatureGen::name() << CMA
      << EmptyObservedFeatureGen::name() << CMA
      << BergsmaKondrakWordFeatureGen::name() << CMA
      << KlementievRothWordFeatureGen::name() << CMA
      << KlementievRothSentenceFeatureGen::name() << "}";      
  stringstream modelMsgObs;
  modelMsgObs << "model {" <<
      StringEditModel<AlignmentHypergraph>::name() << "}";
  stringstream objMsgObs;
  objMsgObs << "objective function {" << LogLinearBinary::name() << CMA <<
      LogLinearBinaryUnscaled::name() << CMA <<
      LogLinearBinaryObs::name() << CMA << LogLinearMulti::name() << CMA <<
      LogLinearMultiELFV::name() << CMA << LogLinearMultiELFV_sigmoid::name() <<
      CMA << LogLinearMultiUW::name() << CMA << MaxMarginBinary::name() << CMA <<
      MaxMarginBinaryObs::name() << CMA << MaxMarginMulti::name() << "}";
  stringstream optMsgObs;
  optMsgObs << "optimization algorithm {" << optAuto << CMA <<
      BmrmOptimizer::name() << CMA << LbfgsOptimizer::name() << CMA <<
      StochasticGradientOptimizer::name() << "}";
  stringstream readerMsg;
  readerMsg << "reader that parses lines from input file {" <<
      CognatePairReader::name() << CMA <<
      CognatePairAlignerReader::name() << CMA <<
      SentencePairReader::name() << CMA <<
      WordPairReader::name() << "}";  
  stringstream regMsg;
  regMsg << "type of regularization {" << RegularizerL2::name() << CMA <<
      RegularizerSoftTying::name() << CMA << RegularizerNone::name() << "}";
  
  opt::options_description options("Main options");
  options.add_options()
    ("add-begin-end", opt::bool_switch(&addBeginEndMarkers), "add begin-/end-\
of-sequence markers to the examples")
    ("beta,b", opt::value<vector<double> >(&betas)->default_value(
        vector<double>(1, 1.0), "1.0"), "the L2 regularization constant used \
in the training objective, i.e., (beta/2)*||w||^2")
    ("directory", opt::value<string>(&dirPath),
        "directory in which to store results (must already exist)")
//  ("em", opt::bool_switch(&optEM), "employ the optimizer inside an EM loop")
    ("eval", opt::value<vector<string> >(&evalFilenames),
        "evaluation data file")
    ("fgen-latent", opt::value<string>(&fgenNameLat)->default_value(
        EmptyAlignmentFeatureGen::name()), fgenMsgLat.str().c_str())
    ("fgen-observed", opt::value<string>(&fgenNameObs)->default_value(
        EmptyObservedFeatureGen::name()), fgenMsgObs.str().c_str())
    ("keep-all-positives", opt::bool_switch(&keepAllPositives),
        "if --sample-train is enabled, this option ensures that all the \
positive examples present in the data are retained")
    ("load-features", opt::value<string>(&loadFeaturesFilename),
        "load features/alphabet from the given file")
    ("model", opt::value<string>(&modelName)->default_value(
        StringEditModel<AlignmentHypergraph>::name()), modelMsgObs.str().c_str())
    ("no-early-grid-stop", opt::bool_switch(&noEarlyGridStop),
        "by default, we break from the grid search loop (over the tolerance \
and beta values) if the optimizer failed to converge; however, if this flag is \
present, all points on the grid will be visited")
    ("no-training", opt::bool_switch(&skipTraining), "load and evaluate \
any existing weight vectors, but don't train at any new points in the grid")
    ("objective", opt::value<string>(&objName)->default_value(
        LogLinearMulti::name()), objMsgObs.str().c_str())
    ("optimizer", opt::value<string>(&optName)->default_value(optAuto),
        optMsgObs.str().c_str())
    ("print-alignments", opt::bool_switch(&printAlignments), "print the max-\
scoring alignment for each eval example to a file (requires --eval); note: \
this operation is relatively slow, since it does not make use of multi-\
threading or fst caching")
    ("reader", opt::value<string>(&readerName), readerMsg.str().c_str())
    ("regularizer", opt::value<string>(&regName)->default_value(
        RegularizerL2::name()), regMsg.str().c_str())
    ("sample-negative-ratio", opt::value<double>(&negativeRatio),
        "for each positive example in the training set, sample this number of \
negative examples (implies --keep-all-positives)")
    ("sample-train", opt::value<double>(&trainFraction),
        "learn on this fraction of the train data (uniformly sampled, \
without replacement); if greater than 1, the value is interpreted as the \
*number* of training examples to sample")
    ("save-features", opt::value<string>(&saveFeaturesFilename),
        "save features/alphabet to the given file")
    ("seed", opt::value<int>(&seed)->default_value(0),
        "seed for random number generator")
    ("split", opt::bool_switch(&split), "used in combination with \
sample-train to use the unselected training examples as the eval set")
    ("threads", opt::value<size_t>(&threads)->default_value(1),
        "number of threads employed to parallelize computations")
    ("tolerance,t", opt::value<vector<double> >(&tolerances)->default_value(
        vector<double>(1, 1e-4), "1e-4"), "the tolerance of the stopping \
criterion used by the optimizer")
    ("train", opt::value<string>(&trainFilename), "training data file")
    ("validation", opt::value<string>(&validationFilename),
        "validation data file (only used by certain optimizers)")
    ("weights-init", opt::value<string>(&weightsInit)->default_value("noise"),
        "initialize weights {heuristic, heuristic+noise, noise, zero}")
    ("weights-noise-level", opt::value<double>(&weightsNoise)->default_value(
        0.01), "Gaussian variance parameter used when applying noise to \
initial weights")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  const bool help = vm.count("help");
  const bool writeFiles = vm.count("directory");
  const bool trainFileSpecified = vm.count("train");
  const bool evalFileSpecified = vm.count("eval");
  const bool validationFileSpecified = vm.count("validation");
  const bool loadFeatures = loadFeaturesFilename.size() > 0;
  const bool saveFeatures = saveFeaturesFilename.size() > 0;
  
  if (!trainFileSpecified && !evalFileSpecified) {
      cout << "Invalid arguments: Either --train or --eval is required\n"
        << options << endl;
    return 1;
  }
  
  if (split && trainFraction == 1.0) {
    cout << "Invalid arguments: Can't have --split with --sample-train=1.0\n"
        << options << endl;
    return 1;
  }
  
  if (istarts_with(objName, "MaxMargin") && regName != RegularizerL2::name()) {
    cout << "Invalid arguments: MaxMargin objectives must use L2 " <<
        "regularization at this time.\n" << options << endl;
    return 1;
  }
  
  if (fgenNameLat == EmptyAlignmentFeatureGen::name() &&
      fgenNameObs == EmptyObservedFeatureGen::name()) {
    cout << "Invalid arguments: fgen-latent and fgen-observed are both set " <<
        "to empty feature generators.\n";
    return 1;
  }
  
  if (loadFeatures && saveFeatures) {
    cout << "Invalid arguments: Can't use --load-features and " <<
        "--save-features at the same time.\n";
    return 1;
  }
  
  if (!iends_with(dirPath, "/"))
    dirPath += "/";
    
  const string alphabetFname = dirPath + "alphabet.txt";
  bool cachingEnabled = false;
  bool resumed = false; // Are we resuming an incomplete run?
  vector<Model*> models;
  
  shared_ptr<Alphabet> loadedAlphabet(new Alphabet(false, false));
  if (filesystem::exists(alphabetFname)) {
    if (loadFeatures || saveFeatures) {
      cout << "Error: Can't load or save features when the default " <<
          "named alphabet file (" << alphabetFname << ") already exists\n";
      return 1;
    }
    if (!loadedAlphabet->read(alphabetFname)) {
      cout << "Error: Unable to read " << alphabetFname << endl;
      return 1;
    }
    resumed = true;
    cout << "Warning: Found existing output files in " << dirPath <<
        ", so treating this as a resumed run. Any evaluation output files " <<
        "will be overwritten.\n";
  }

  for (size_t th = 0; th < threads; th++) {
    shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
    if (resumed)
      alphabet = loadedAlphabet;
      
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
    else if (fgenNameObs == BergsmaKondrakWordFeatureGen::name())
      fgenObs.reset(new BergsmaKondrakWordFeatureGen(alphabet));
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
    if (modelName == StringEditModel<AlignmentHypergraph>::name()) {
      model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
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
  
  timer::auto_cpu_timer timerTotal;
  
  Dataset trainData(threads);
  Dataset evalData(threads);
  
  // initialize the input reader
  scoped_ptr<InputReader> reader;
  if (readerName == CognatePairReader::name())
    reader.reset(new CognatePairReader());
  else if (readerName == CognatePairAlignerReader::name())
    reader.reset(new CognatePairAlignerReader());
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
  if (reader)
    reader->setAddBeginEndMarkers(addBeginEndMarkers);
  
  if (!help && trainFileSpecified) {
    {
      cout << "Loading " << trainFilename << " ...\n";
      timer::auto_cpu_timer loadTrainTimer;
      if (Utility::loadDataset(*reader, trainFilename, trainData)) {
        cout << "Error: Unable to load train file " << trainFilename << endl;
        return 1;
      }
      cout << "Read " << trainData.numExamples() << " train examples, " <<
          trainData.getLabelSet().size() << " classes\n";
    }
  
    if (negativeRatio > 0.0 || trainFraction != 1.0) {
      if (negativeRatio > 0.0)
        keepAllPositives = true; // keepAllPositives is implied in this case
      mt19937 mt(seed);
      uniform_int<> uni(0, trainData.numExamples()-1);
      variate_generator<mt19937, uniform_int<> > rgen(mt, uni);
      set<int> selected;
      if (keepAllPositives) {
        for (size_t i = 0; i < trainData.numExamples(); i++) {
          if (trainData.getExamples()[i].y() == TrainingObjective::kPositive)
            selected.insert((int)i);
        }
        cout << "Keeping all " << selected.size() << " positive examples.\n";
      }
      size_t subTrainSize;      
      if (negativeRatio > 0.0) {
        subTrainSize = (size_t)(selected.size() * negativeRatio) +
          selected.size();
      }
      else {
        assert(trainFraction != 1.0);
        if (trainFraction < 1)
          subTrainSize = (size_t)(trainFraction * trainData.numExamples());
        else // interpret trainFraction as the number of examples requested
          subTrainSize = (size_t)trainFraction;
      }
      if (subTrainSize >= trainData.numExamples()) {
        cout << "Invalid arguments: There are not enough training examples " <<
            "for the sampling options that you specified.\n";
        return 1;
      }
      if (selected.size() >= subTrainSize) {
        cout << "Invalid arguments: There are more positive examples than " <<
            "the total number of training examples you asked to sample. " <<
            "Please lower the value of --sample-train or remove the " <<
            "--sample-all-positives flag.\n";
        return 1;
      }
      cout << "Randomly selecting " << (subTrainSize - selected.size()) <<
          (keepAllPositives ? " negative" : " train") << " examples.\n";
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
  else if (objName == LogLinearMultiELFV::name())
    objective.reset(new LogLinearMultiELFV(trainData, models));
  else if (objName == LogLinearMultiELFV_sigmoid::name())
    objective.reset(new LogLinearMultiELFV_sigmoid(trainData, models));
  else if (objName == LogLinearMultiUW::name())
    objective.reset(new LogLinearMultiUW(trainData, models));
  else if (objName == LogLinearBinary::name())
    objective.reset(new LogLinearBinary(trainData, models));
  else if (objName == LogLinearBinaryUnscaled::name())
    objective.reset(new LogLinearBinaryUnscaled(trainData, models));
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

  // Initialize the regularizer.
  shared_ptr<Regularizer> regularizer;
  if (regName == RegularizerNone::name())
    regularizer.reset(new RegularizerNone());
  else if (regName == RegularizerL2::name())
    regularizer.reset(new RegularizerL2());
  else if (regName == RegularizerSoftTying::name())
    regularizer.reset(new RegularizerSoftTying());
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized regularizer name was given: "
          << regName << endl << options << endl;
      return 1;
    }
  }
  if (regularizer->processOptions(argc, argv)) {
    cout << "Regularizer::processOptions() failed." << endl;
    return 1;
  }

  // Initialize the optimizer.
  shared_ptr<Optimizer> optInner;
  if (optName == optAuto) {
    // Automatically select an appropriate optimizer for the chosen objective.
    if (istarts_with(objName, "LogLinear"))
      optInner.reset(new LbfgsOptimizer(objective, regularizer));
    else if (istarts_with(objName, "MaxMargin"))
      optInner.reset(new BmrmOptimizer(objective, regularizer));
    else {
      cout << "Automatic optimizer selection failed for " << objName << endl;
      return 1;
    }
  }
  else if (optName == LbfgsOptimizer::name())    
    optInner.reset(new LbfgsOptimizer(objective, regularizer));
  else if (optName == BmrmOptimizer::name())
    optInner.reset(new BmrmOptimizer(objective, regularizer));
  else if (optName == StochasticGradientOptimizer::name())
    optInner.reset(new StochasticGradientOptimizer(objective, regularizer));
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized optimizer name was given: "
          << optName << endl << options << endl;
      return 1;
    }
  }
  if (optInner->processOptions(argc, argv)) {
    cout << "Optimizer::processOptions() failed." << endl;
    return 1;
  }
  
  boost::shared_ptr<Dataset> validationData;
  shared_ptr<ValidationSetHandler> vsh;
  if (!help && validationFileSpecified) {
    // Enumerate the examples in the validation set such that the ids do not
    // overlap with the training set.
    const size_t nextId = trainData.getExamples()[trainData.numExamples()-1].
        x()->getId() + 1;
    validationData.reset(new Dataset(threads));
    cout << "Loading " << validationFilename << " ...\n";
    timer::auto_cpu_timer loadValidationTimer;
    if (Utility::loadDataset(*reader, validationFilename, *validationData,
        nextId)) {
      cout << "Error: Unable to load validation file " << validationFilename <<
          endl;
      return 1;
    }
    assert(nextId == validationData->getExamples()[0].x()->getId());
    cout << "Read " << validationData->numExamples() << " validation examples, "
        << validationData->getLabelSet().size() << " classes\n";
    vsh.reset(new ValidationSetHandler(validationData, objective));
    if (vsh->processOptions(argc, argv)) {
      cout << "ValidationSetHandler::processOptions() failed." << endl;
      return 1;
    }
    optInner->setValidationSetHandler(vsh);
  }

  // Wrap the optimizer in an EM procedure if requested.
  shared_ptr<Optimizer> optimizer;
  if (optEM) {
    optimizer.reset(new EmOptimizer(objective, regularizer, optInner));
    if (optimizer->processOptions(argc, argv)) {
      cout << "Optimizer::processOptions() failed." << endl;
      return 1;
    }
  }
  else
    optimizer = optInner; // Note: processOptions has already been called

  if (help) {
    cout << options << endl;
    return 1;
  }
  
  if (!resumed && !loadFeatures) {
    cout << "Gathering features ...\n";
    size_t maxNumFvs = 0, totalNumFvs = 0;
    {
      timer::auto_cpu_timer gatherTimer;
      objective->gatherFeatures(maxNumFvs, totalNumFvs);
      assert(maxNumFvs > 0 && totalNumFvs > 0);
      objective->combineAlphabets(trainData.getLabelSet());
    }
  }
  
  shared_ptr<const AlignmentFeatureGen> fgenLat =
      objective->getModel(0).getFgenLatent();
  shared_ptr<const ObservedFeatureGen> fgenObs =
      objective->getModel(0).getFgenObserved();
  shared_ptr<Alphabet> alphabet = fgenLat->getAlphabet();
  assert(alphabet == fgenObs->getAlphabet()); // Assume the alphabet is shared

  if (loadFeatures) {
    if (!filesystem::exists(loadFeaturesFilename)) {
      cout << "Error: " << loadFeaturesFilename << " does not exist.\n";
      return 1;
    }
    if (!alphabet->read(loadFeaturesFilename)) {
      cout << "Error: Unable to read " << loadFeaturesFilename << endl;
      return 1;
    }
    cout << "Loaded features from " << loadFeaturesFilename << endl;
    alphabet = objective->combineAlphabets(trainData.getLabelSet());
  }
  
  if (alphabet->size() == 0) {
    cout << "Error: No features were found!\n";
    return 1;
  }
  cout << "Extracted " << alphabet->size() << " features\n";  
  
  if (saveFeatures) {
    alphabet->lock();
    if (!alphabet->write(saveFeaturesFilename)) {
      cout << "Warning: Unable to write " << saveFeaturesFilename << endl;
      return 1;
    }
    else {
      cout << "Saved features to " << saveFeaturesFilename << endl;
      return 0;
    }
  }

  // Enable caching at this point, if requested.
  if (cachingEnabled) {
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      objective->getModel(i).setCacheEnabled(true);
      // The cache may contain a "reusable" fst: see StringEditModel::getFst().
      objective->getModel(i).emptyCache();
      if (optimizer->isOnline() && validationData->numExamples() > 0) {
        // If we're employing an online learner (e.g., SGD), we don't want to
        // cache graphs for training examples because presumably the dataset is
        // very large. In this case, if caching is enabled, we interpret it to
        // mean that we want to cache the validation set examples, which will
        // be classified multiple times during training.
        objective->getModel(i).onlyCacheIdsGreaterThanOrEqualTo(
            validationData->getExamples()[0].x()->getId());
      }
    }
  }
  
  assert(betas.size() > 0);
  assert(tolerances.size() > 0);
  
  // Set the initial parameters.
  Parameters theta0 = objective->getDefaultParameters(alphabet->size());
  // Note: The call to setupParameters may modify theta0 and alphabet, such
  // that, e.g., alphabet->size() may subsequently return a different value.
  // FIXME: We cannot add another dummy label if an alphabet was loaded that
  // already had a dummy label added to it.
  if (resumed) {
    cout << "Error: Resuming an experiment is not supported in this version.\n";
    return 1;
  }
  // A label that is "instantiated" has at least one feature associated with it.
  // Specifically, for a binary objective, the negative class will not be
  // instantiated.
  set<Label> instantiatedLabels;
  if (objective->isBinary())
    instantiatedLabels.insert(TrainingObjective::kPositive);
  else
    instantiatedLabels = trainData.getLabelSet();
  regularizer->setupParameters(theta0, *alphabet, instantiatedLabels,
      seed);
  alphabet->lock(); // We can lock the Alphabet at this point.
  initWeights(theta0.w, weightsInit, weightsNoise, seed, alphabet,
      instantiatedLabels, fgenLat);
  if (theta0.hasU()) {
    // Note: We modify the seed to avoid symmetry in the parameters.
    initWeights(theta0.u, weightsInit, weightsNoise, seed + 1, alphabet,
        instantiatedLabels, fgenLat);
  }
  
  // If an output directory was specfied, save the alphabet and options.
  if (writeFiles && !resumed) {
    if (!alphabet->write(alphabetFname))
      cout << "Warning: Unable to write " << alphabetFname << endl;  
    const string optsFname = dirPath + "options.txt";
    ofstream optsOut(optsFname.c_str());
    if (optsOut.good()) {
      optsOut << optsStream.str();
      optsOut.close();
    }
    else
      cout << "Warning: Unable to write " << optsFname << endl;
  }

  // Train weights for each combination of the beta and tolerance parameters.
  // Note that the fsts (if caching is enabled) will be reused after being
  // built during the first parameter combination.
  vector<Parameters> weightVectors;
  sort(tolerances.rbegin(), tolerances.rend()); // sort in descending order
  sort(betas.rbegin(), betas.rend()); // sort in descending order
  BOOST_FOREACH(const double tol, tolerances) {
    BOOST_FOREACH(const double beta, betas) {    
      assert(beta > 0); // by definition, these should be positive values
      assert(tol > 0);
      
      weightVectors.push_back(objective->getDefaultParameters(
          alphabet->size()));

      Parameters& theta = weightVectors.back();
      theta.setParams(theta0); // FIXME: This line is causing an assert failure
      assert(weightVectors.size() > 0);
      const size_t wvIndex = weightVectors.size() - 1;
      
      stringstream weightsFnameW, weightsFnameU;
      weightsFnameW << dirPath << wvIndex << "-weights.txt";
      weightsFnameU << dirPath << wvIndex << "-weightsU.txt";
      bool weightsFileIsGood = false;
      
      if (filesystem::exists(weightsFnameW.str()))
      {
        assert(resumed);
        if (theta.w.read(weightsFnameW.str(), alphabet->size()))
          weightsFileIsGood = true;
        else
          cout << "Warning: Unable to read " << weightsFnameW.str() << endl;
          
        if (weightsFileIsGood && theta.hasU())
        {
          if (!theta.u.read(weightsFnameU.str(), alphabet->size())) {
            cout << "Warning: Unable to read " << weightsFnameU.str() << endl;
            weightsFileIsGood = false;
          }
        }
      }      
      
      if (!weightsFileIsGood) {
        if (trainFileSpecified && !skipTraining) {
          // Train the model.
          Optimizer::status status = Optimizer::FAILURE;
          double fval = 0.0;
          cout << "Calling Optimizer.train() with beta=" << beta << " and " <<
              "tolerance=" << tol << endl;
          {
            timer::auto_cpu_timer trainTimer;
            regularizer->setBeta(beta);
            status = optimizer->train(theta, fval, tol);
          }
          // Update: Besides the cost of evaluation, there's no drawback to
          // keeping classifiers for which the optimizer didn't report
          // convergence. The status still gets reported below.
//        if (status == Optimizer::FAILURE) {
//          cout << "Warning: Optimizer returned status " << status << ". " <<
//              "Discarding classifier.\n";
//          weightVectors.pop_back();
//          continue;
//        }
          if (!noEarlyGridStop && status == Optimizer::MAX_ITERS_CONVEX) {
            cout << "Warning: Optimizer returned status " << status << ". " <<
                "Discarding classifier and skipping to next tolerance value.\n";
            weightVectors.pop_back();
            break;
          }
          cout << wvIndex << "-status: " << status << endl;
          cout << wvIndex << "-Objective-Value: " << fval << endl;
        }
        else {
          // If no train file was specified or the user requested that training
          // be skipped, we evaluate only the existing weight vectors in the
          // given directory (but do not train any new ones, even if there are
          // points on the grid that have not been successfully tried).
          cout << "Warning: There will be no results for beta=" << beta <<
              " and tolerance=" << tol << endl;
          weightVectors.pop_back();
          break;
        }
      }
      
      cout << wvIndex << "-beta: " << beta << endl;
      cout << wvIndex << "-tolerance: " << tol << endl;

      // If an output directory was specfied, save the weight vector.
      if (writeFiles && !weightsFileIsGood) {
        if (!theta.w.write(weightsFnameW.str()))
          cout << "Warning: Unable to write " << weightsFnameW.str() << endl;
        if (!theta.u.write(weightsFnameU.str()))
          cout << "Warning: Unable to write " << weightsFnameU.str() << endl;
      }
      
      // Classify train examples and optionally write the predictions to a file.
      // Note: The model's transducer cache can still be used at this point, so
      // we defer purging it until after classifying the train examples.
      stringstream predictFname; // Defaults to empty string (for evaluate()).
      if (writeFiles)
        predictFname << dirPath << wvIndex << "-train_predictions.txt";
      if (!filesystem::exists(predictFname.str())) {
        cout << "Classifying train examples ...\n";
        {
          stringstream identifier;
          identifier << wvIndex << "-Train";
          timer::auto_cpu_timer classifyTrainTimer;
          Utility::evaluate(theta, objective, trainData, identifier.str(),
              predictFname.str());
        }
      }
    }
  }
  
  trainData.clear(); // We no longer need the training data.

  // Clear the fsts that were cached for the training data.
  if (cachingEnabled) {
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      Model& model = objective->getModel(i);
      model.setCacheEnabled(false);
      model.emptyCache();
    }
  }
  
  // Load the eval examples from the specified file.
  if (!split && evalFileSpecified && weightVectors.size() > 0) {
    int evalId = 0;
    BOOST_FOREACH(const string& evalFilename, evalFilenames) {
      cout << "Loading " << evalFilename << " ...\n";
      timer::auto_cpu_timer loadEvalTimer;
      Dataset evalData(threads); // Intentionally shadowing previous variable.
      if (Utility::loadDataset(*reader, evalFilename, evalData)) {
        cout << "Error: Unable to load eval file " << evalFilename << endl;
        return 1;
      }
      cout << "Read " << evalData.numExamples() << " eval examples, " <<
          evalData.getLabelSet().size() << " classes\n";
          
      // This situation can arise if, e.g., the eval set only contains examples
      // from one class.
      if (evalData.getLabelSet().size() < trainData.getLabelSet().size()) {
        evalData.addLabels(trainData.getLabelSet());
        cout << "The eval set has fewer classes than train. Adding labels from "
            << "train set to eval set. Number of classes is now "
            << evalData.getLabelSet().size() << ".\n";
      }
      assert(evalData.numExamples() > 0);
      assert(evalData.getLabelSet().size() > 1);
      
      evaluateMultipleWeightVectors(weightVectors, evalData, objective,
          dirPath, evalId++, writeFiles, printAlignments, cachingEnabled);
    }
  }
  else if (weightVectors.size() == 0) {
    cout << "Warning: No classifiers were successfully trained; therefore, "
        << "no evaluation will be performed. Perhaps the range of the "
        << "tolerance and/or beta parameters should be adjusted. Or, you may "
        << "re-run the experiment with the --no-early-grid-stop flag.\n";
  }

  return 0;
}
