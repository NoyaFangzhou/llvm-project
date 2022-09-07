//
// Created by noya-fangzhou on 7/15/22.
//

#ifndef LLVM_MODELCODEGENPRO_H
#define LLVM_MODELCODEGENPRO_H
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "PlussUtils.h"
#include "BranchAnalysis.h"
#include "LoopAnalysisUtils.h"
#include "SamplerCodeGenUtils.h"
#include "AccessGraphAnalysisUtils.h"

namespace ModelCodeGenPro {

enum ReuseType {
  FULL_SHARE,
  SPATIAL_SHARE,
  OTHER
};

struct ModelCodeGenProWrapperPass : FunctionPass {

  static char ID;
  ModelCodeGenProWrapperPass();

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool EnableSampling;
  SamplingType SamplingMethod;
  SampleNumberAnalyzer *Analyzer;
  SamplerCodeGenerator *CodeGen;
  unordered_map<PHINode *, string> InductionVarNameTable;
  unordered_set<string> ArrayNames;
  unordered_set<LoopTNode *> ModelValidateLoops;
  DenseMap<SPSTNode *, AccGraph *> Graphs;
  DenseMap<SPSTNode *, LoopTNode *> ParentMappingInAbstractionTree;
  vector<LoopTNode *> LoopNodes;
  DenseMap<RefTNode *, string> PointerAliasPairs;
  DenseMap<SPSTNode *, string> SPSNodeNameTable;
  // ordered the referece node with their topology order
  DenseMap<RefTNode *, int> RefNodePriority;
  DenseMap<RefTNode *, unsigned> PerReferenceSampleCnt;
  RefTNode *FirstAccess;

  TreeStructureQueryEngine *QueryEngine;
  StringTranslator *Translator;

  ScalarEvolutionWrapperPass *SEWP;
  BranchAnalysis::BranchAnalysisWrapperPass *BAWP;

  void ModelUtilGen();
  void HeaderGen();
  void MainFuncGen();
  void SamplerBodyGen(bool);
  void RefAddressFuncGen();

  // whole program sampler codegen
  void SamplerBodyGenImpl(SPSTNode *);
  void LoopIterUpdateGen(AccGraph *G, AccGraphEdge *E, LoopTNode *IDomLoop,
                         bool EnableIterationInc, bool EnableIterationUpdate);

  void ParallelSamplerBodyGenImpl(SPSTNode *, AccGraph *,
                                  Interleaving type = UNIFORM_INTERLEAVING);
  void ParallelSamplerBranchBodyGenImpl(DummyTNode *, Interleaving, SPSTNode *);
  // Per reference sampler codegen
  void ParallelSamplerSamplingHeaderGen(LoopTNode *, AccGraph *, RefTNode *,
                                        Interleaving, SchedulingType);
  void PerRefSamplerBodyGenImpl(SPSTNode *, RefTNode *TargetRef);
  void
  PerRefParallelSamplerBodyGenImpl(SPSTNode *, RefTNode *TargetRef, AccGraph *,
                                   Interleaving type = UNIFORM_INTERLEAVING);
  void PerRefParallelSamplerBranchBodyGenImpl(DummyTNode *, Interleaving,
                                              SPSTNode *, RefTNode *);

  void SampleIterationCodeGen(RefTNode *, bool);

  // Model
  unordered_map<string, SPSTNode *> RefNameToNodeMapping;

  // Utils
  string EmitLoopNodeExpr(LoopTNode *, bool isSampledLoop = false);
  string EmitLoopTripExpr(LoopTNode *);
  string EmitBranchCondExpr(BranchTNode *);
  string EmitOneParallelIterationTripFormula(LoopTNode *);
  string EmitTripFormula(SPSTNode *);
  bool canBeSimplifiedByFormula(SPSTNode *);
  string EmitRefNodeAccessExprAtIdx(Value *, bool isInParallel = false);
  vector<string> EmitRefNodeAccessExpr(RefTNode *, bool isInParallel = false);
  // the first element is not null if has LOWERBOUND_DEPENDENCY, the second element is not null if has UPPERBOUND_DEPENDENCY
  pair<LoopTNode *, LoopTNode *> CheckDependenciesInPath(LoopTNode *,
                                                         AccPath &);
  // check if the given loop node's induction variable has dependency child
  // or not. It will be used to determine whether we should use
  // OpenMP dynamic scheduling
  bool hasInductionVarDependenceChildren(LoopTNode *);
  bool hasParallelLoopInductionVar(Value *);
  bool isInParallelLoop(RefTNode *);

  string ReplaceInductionVarExprByType(Value *, CodeGenType type,
                                       TranslateStatus &);
  string RepresentLoopTrip(LoopTNode *);

  // return -1 if given phinode is an induction variable ouside the loop nests
  // otherwise, return its distance to the parallel loop
  int isInsideParallelLoopNest(PHINode *);
  int ComputeDistanceToParallelLoop(LoopTNode *);

  // return the reuse type of the given RefNode
  // FULL_SHARE: all its indices do not contain the induction variable
  // of the parallel loop,
  // e.g.
  // #pragma omp parallel for
  // for (i in 0 ... NI)
  //  for (j in 0 ... NJ)
  //    for (k in 0 ... NK)
  //      A[j][k] = B[j]
  // PARTIAL_SHARE: the parallel loop induction variable is the last indices
  // e.g.
  // #pragma omp parallel for
  // for (i in 0 ... NI)
  //  for (j in 0 ... NJ)
  //    for (k in 0 ... NK)
  //      A[j][i] = B[i]
  ReuseType isInterThreadSharingRef(RefTNode *);
};
} // end of namespace
#endif // LLVM_MODELCODEGENPRO_H
