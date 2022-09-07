//
//  LoopAnalysisWrapperPass.hpp
//  LLVM
//
//  This pass analyze the loop and build the loop abstract
//  tree of the program
//
//  Created by noya-fangzhou on 10/13/21.
//

#ifndef LOOPANALYSIS_H
#define LOOPANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "BranchAnalysis.h"
#include "LoopAnalysisUtils.h"
#include "PlussUtils.h"

using namespace llvm;
using namespace std;

namespace PlussLoopAnalysis {
struct LoopAnalysisWrapperPass : public FunctionPass {
  static char ID;
  LoopAnalysisWrapperPass();

  // The dummy root of the Abstraction Tree
  SPSTNode *TreeRoot;

  // All pairs that has dependences, the key is depens on the value in this
  // map
  SmallSet<IVDepNode *, 16> InductionVarDepForest;

  void FindAllInductionVarDependencyInLoop(Loop *L);

  void DumpTree();
  void DumpInductionVarDependency();
  // Write the tree in DOT code.
  // By copy-past the code in the output file, one can view the abstraction
  // tree on the online graphiz website.
  // i.e. https://dreampuf.github.io/GraphvizOnline
  void ViewLoopTree(SPSTNode *Root, string title);
  void ViewIndudctionVarDependency();
  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  unsigned LoopCnt;
  unordered_map<Instruction *, RefTNode *> AccessToNodeMapping;
  unordered_map<Loop *, LoopTNode *> LoopToNodeMapping;
  StringTranslator *Translator;
  LoopInfoWrapperPass *LIWP;
  ScalarEvolutionWrapperPass *SEWP;
  BranchAnalysis::BranchAnalysisWrapperPass *BAWP;
  void AddRefNodeToList(BasicBlock *Block, vector<SPSTNode *> &SPSNodeList);
  SPSTNode *BuildTreeForLoopImpl(Loop *,
                                 unordered_set<BasicBlock *> &,
                                 unsigned LoopLevel = 0);
  SPSTNode *BuildBranchNodeFor(BranchInst *);
  void AppendBranchNodesOnTreeImpl(SPSTNode *);
  void AppendRefNodesOnTreeImpl(SPSTNode *);
  void DecorateBranchNode(BranchTNode *BrNode, BasicBlock *TakenBranchHeader,
                                        BasicBlock *TakenBranchTerminator,
                          bool isTrue= true);
  void GenerateRefExpr(RefTNode *Node);
  void GenerateLoopExpr(LoopTNode *Node);
  void GenerateBranchExpr(BranchTNode *Node);
  bool IsPlussParallelLoop(Loop *L);
  void BuildParentMapping(SPSTNode *);

  DIRECTION hasDependency(Loop *Parent, Loop *Child);
};


} // end of LoopAnalysisWrapperPass namespace


#endif /* LOOPANALYSIS_H */
