//
// Created by noya-fangzhou on 4/20/22.
//

#ifndef LLVM_REFERENCEANALYSIS_H
#define LLVM_REFERENCEANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "PlussUtils.h"
#include "BranchAnalysis.h"
#include "LoopAnalysisUtils.h"
#include "SamplerCodeGenUtils.h"
#include "AccessGraphAnalysisUtils.h"

namespace ReferenceAnalysis {

struct ReferenceAnalysis : FunctionPass {

  static char ID;
  ReferenceAnalysis();

  unordered_map<LoopTNode *, unordered_set<string>> PerLoopArraysMap;
  unordered_map<RefTNode*, unsigned> ReferenceDependenceCount;

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  TreeStructureQueryEngine *QueryEngine;
  StringTranslator *Translator;

  ScalarEvolutionWrapperPass *SEWP;
  BranchAnalysis::BranchAnalysisWrapperPass *BAWP;

  DependenceInfo *DI;

  SPSTNode *TreeRoot;

  void ReferenceDependenceAnalysis(LoopTNode *);
  /* Get the number of references who can form a reuse with the
   * given reference. References should be inside the same loop
   */
  unsigned GetNumberOfDependenceReferences(RefTNode*);

}; // end of ReferenceAnalysis struct


} // end of namespace ReferenceAnalysis



#endif // LLVM_REFERENCEANALYSIS_H
