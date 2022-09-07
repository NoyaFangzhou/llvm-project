//
// Created by noya-fangzhou on 1/21/22.
//

#ifndef LLVM_ITERATIONCOUNTERCODEGEN_H
#define LLVM_ITERATIONCOUNTERCODEGEN_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "PlussUtils.h"
#include "BranchAnalysis.h"
#include "LoopAnalysisUtils.h"
#include "SamplerCodeGenUtils.h"
#include "AccessGraphAnalysisUtils.h"

namespace IterationCounterCodeGen {
struct IterationCountCodeGenWrapperPass : FunctionPass {

  static char ID;
  IterationCountCodeGenWrapperPass();

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  SamplerCodeGenerator *CodeGen;
  unordered_map<PHINode *, string> InductionVarNameTable;
  unordered_set<LoopTNode *> ModelValidateLoops;
  DenseMap<SPSTNode *, AccGraph *> Graphs;
  DenseMap<SPSTNode *, LoopTNode *> ParentMappingInAbstractionTree;
  vector<LoopTNode *> LoopNodes;
  DenseMap<RefTNode *, string> PointerAliasPairs;
  DenseMap<SPSTNode *, string> SPSNodeNameTable;
  DenseMap<RefTNode *, unsigned > PerReferenceSampleCnt;

  TreeStructureQueryEngine *QueryEngine;
  StringTranslator *Translator;

  ScalarEvolutionWrapperPass *SEWP;
  BranchAnalysis::BranchAnalysisWrapperPass *BAWP;

  void HeaderGen();
  void MainFuncGen();
  void CounterBodyGen();

  // whole program sampler codegen
  void CounterBodyGenImpl(SPSTNode *);

  // Utils
  string EmitLoopNodeExpr(LoopTNode *, bool isSampledLoop=false);
  string EmitLoopTripExpr(LoopTNode *);
  string EmitBranchCondExpr(BranchTNode *);
  vector<string> EmitRefNodeAccessExpr(RefTNode *, bool isInParallel=false);
  bool isInParallelLoop(RefTNode *);
};
} // end of IterationCounterCodeGen namespace

#endif // LLVM_ITERATIONCOUNTERCODEGEN_H
