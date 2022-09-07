//
// Created by noya-fangzhou on 11/17/21.
//

#ifndef LLVM_PLUSSABSTRACTIONTREEANALYSIS_H
#define LLVM_PLUSSABSTRACTIONTREEANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "PlussUtils.h"
#include "LoopAnalysisUtils.h"



// generate ID of each reference node and branch node

namespace TreeAnalysis {

struct PlussAbstractionTreeAnalysis : public FunctionPass {
  static char ID;
  PlussAbstractionTreeAnalysis();

  DenseMap<SPSTNode *, LoopTNode *> ImmediateParentMapping;

  vector<LoopTNode *> LoopNodeList;

  // key can be RefTNode/BranchTNode type
  DenseMap<SPSTNode *, string> NodeToRefNameMapping;

  // base to base mappinp. key, value are the array base instructions.
  DenseMap<RefTNode *, string> PointerAliasCandidate;

  unordered_map<SPSTNode *, unordered_set<RefTNode *>> ReachableAnalysisResult;

  unordered_map<string, vector<string>> ArrayBound;

  vector<RefTNode *> RefNodeOrderedList;

  TreeStructureQueryEngine &getQueryEngine();

  /* Analysis pass main function */
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  AliasAnalysis *AA;
  // The dummy root of the Abstraction Tree
  SPSTNode *TreeRoot;
  StringTranslator *Translator;
  unsigned BranchCount; // generate ID for each branch

  TreeStructureQueryEngine *QueryEngine;
  // key is the base of each RefTNode
  unordered_map<Value *, unsigned> PerArrayReferenceCount; // generate ID for each array reference
  void GenerateReferenceName();
  void GenerateReferenceNameImpl(SPSTNode *);
  void BuildParentMapping(SPSTNode *);
  void BuildPointerAliasMap();
  void DoReferenceReachableAnalysis();
  void FindInductionVarBound(); // used to compute the address for each array
  PHINode *GetInductionVarFromArraySubscript(Value *);
};
} // end of TreeAnalysis namespace


#endif // LLVM_PLUSSABSTRACTIONTREEANALYSIS_H
