//
// Created by noya-fangzhou on 11/13/21.
//

#ifndef LLVM_ACCESSGRAPHANALYSIS_H
#define LLVM_ACCESSGRAPHANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "BranchAnalysis.h"
#include "LoopAnalysisUtils.h"
#include "AccessGraphAnalysisUtils.h"
#include "PlussUtils.h"

namespace AccessGraphAnalysis {

struct AccessGraphAnalysisPass : public FunctionPass {
  static char ID;
  AccessGraphAnalysisPass();

  DenseMap<SPSTNode *, AccGraph *> TopNodeToGraphMapping;

  void ViewGraph(AccGraph *G, string);
  void BuildAccessGraph(SPSTNode *Root);

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  SPSTNode *TreeRoot;
  PostDominatorTree *PDT;
};

} // end of AccessGraphAnalysis namespace

#endif // LLVM_ACCESSGRAPHANALYSIS_H
