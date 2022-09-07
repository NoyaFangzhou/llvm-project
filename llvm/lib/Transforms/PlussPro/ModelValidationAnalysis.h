//
// Created by noya-fangzhou on 1/6/22.
// This is the pass that analyze how many loop nests can be analyzed by the
// model
//

#ifndef LLVM_MODELVALIDATIONANALYSIS_H
#define LLVM_MODELVALIDATIONANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "PlussUtils.h"
#include "LoopAnalysisUtils.h"

// generate ID of each reference node and branch node

namespace ModelAnalysis {

struct ModelValidationAnalysis : public FunctionPass {
  static char ID;
  ModelValidationAnalysis();

  // all elements inside is a parallel loop
  unordered_set<LoopTNode *> ModelApplicableLoops;

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  // The dummy root of the Abstraction Tree
  SPSTNode *TreeRoot;
  TreeStructureQueryEngine *QueryEngine;
  void analysis();
  void analysisImpl(SPSTNode *, TreeStructureQueryEngine &);
  bool hasInductionVarDependenceChildren(LoopTNode *);
  unsigned CountPotentialReuseReferences(LoopTNode *);
};
} // end of ModelAnalysis namespace


#endif // LLVM_MODELVALIDATIONANALYSIS_H
