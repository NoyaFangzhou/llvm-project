//
// This pass is designed to analyze branch conditions
// it serves two purpose:
//   1. check if this branch condition can be handled by Pluss
//   2. find all BasicBlocks inside a loop that *may* execute because of branch
//      condition
// Created by noya-fangzhou on 11/5/21.
//

#ifndef LLVM_BRANCHANALYSIS_H
#define LLVM_BRANCHANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "PlussUtils.h"
#include "LoopAnalysisUtils.h"

using namespace llvm;
using namespace std;

namespace BranchAnalysis {

typedef SmallVector<BasicBlock *, 16> BranchPath;

class BranchAnalysis {
  ScalarEvolution *SE;
  unordered_map<PHINode *, string> &InductionVarNameTable;
public:
  BranchAnalysis(ScalarEvolution *SE, unordered_map<PHINode *, string> &Table)
      : SE(SE), InductionVarNameTable(Table) {}

  ICmpInst *GetLoopBranch(Loop *L, bool &);

  BranchInst *GetBranchInsideBlock(BasicBlock *BB);
  void GetImmediateBranchInsideLoop(Loop *L,
                                    SmallVectorImpl<BranchInst *> &Branches);

  /// a branch condition that can be handled by Pluss if and only if
  /// all operands in its condition comes from the loop induction variable
  bool IsPlussSupportBranch(Value *Cmp);
private:
  bool IsPlussSupportBranchImpl(Value *V);

};

struct BranchAnalysisWrapperPass : public FunctionPass {
  static char ID;
  BranchAnalysisWrapperPass();

  /// To find all basicblocks inside a branch conditions
  /// We traverse all basicblocks inside a loop, if a basicblock has two parents
  /// it means:
  /// 1. it is a loop branch
  /// 2. it is a branch condition
  void FindBranchBasicBlockInLoop(Loop *L);

  void DumpBranchBasicBlock();

  bool LoopHasBranchInside(Loop *L);

  BranchAnalysis &getBranchAnalyzer() { return *BranchAnalyzer; }

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  PostDominatorTree *PDT;
  BranchAnalysis *BranchAnalyzer;
};

} // end of namespace BranchAnalysis

#endif // LLVM_BRANCHANALYSIS_H
