//
// Created by noya-fangzhou on 11/8/21.
//

#ifndef LLVM_INDUCTIONVARANALYSIS_H
#define LLVM_INDUCTIONVARANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Analysis/LoopInfo.h"
#include "PlussUtils.h"

using namespace llvm;
using namespace std;

extern cl::opt<string> FunctionName;

namespace InductionVarAnalysis {


struct InductionVarAnalysis : public FunctionPass {
  static char ID;
  InductionVarAnalysis();

  unordered_map<PHINode *, string> InductionVarNameTable;

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void PrintInductionVarNameTable();

private:
  LoopInfoWrapperPass *LIWP;
  ScalarEvolutionWrapperPass *SEWP;
  void GenerateInductionVarNameForLoop(Loop *L, unsigned level);
};


} // end of InductionVarAnalysis namespace

#endif // LLVM_INDUCTIONVARANALYSIS_H
