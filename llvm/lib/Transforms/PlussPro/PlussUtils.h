//
// Created by noya-fangzhou on 11/5/21.
//

#ifndef LLVM_PLUSSUTILS_H
#define LLVM_PLUSSUTILS_H

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <stack>
#include <queue>
#include <memory>
#include <algorithm>
#include <sstream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"

using namespace std;
using namespace llvm;

typedef SmallVector<BasicBlock *, 16> Path;

enum SchedulingType {
  STATIC,
  DYNAMIC,
  UNSPECIFIED
};


bool InductionVariableArithmeticInst(Value *, SmallPtrSetImpl<PHINode *> &);

enum TranslateStatus {
  SUCCESS,
  FAIL,
  NOT_TRANSLATEABLE
};

class StringTranslator {
  ScalarEvolution *SE;
  const unordered_map<PHINode *, string> &InductionVarNameTable;
public:
  StringTranslator(ScalarEvolution *SE, unordered_map<PHINode *, string> &Table)
      : SE(SE), InductionVarNameTable(Table) {}
  string SCEVToStringExpr(const SCEV *S, Loop *L, TranslateStatus &status);
  string ValueToStringExpr(Value *V, TranslateStatus &status);
  string ConditionToStringExpr(CmpInst *CI, TranslateStatus &status, bool isDotFormat=true);
  string PredicateToStringExpr(CmpInst::Predicate P, TranslateStatus &status);
};


BasicBlock *ImmediatePostDominator(PostDominatorTree &PDT, BasicBlock *Target);

void FindAllPathesBetweenTwoBlock(BasicBlock *, BasicBlock *,
                                  SmallVectorImpl<Path> &);
void FindAllBasicBocksBetweenTwoBlock(BasicBlock *, BasicBlock *,
                                      SmallVectorImpl<BasicBlock *> &);


PHINode *getInductionVariable(Loop *L, ScalarEvolution &SE);
PHINode *getInductionVariableV2(Loop *L, ScalarEvolution &SE);
bool FindValueInInstruction(Value *, Value *);
bool FindValueInSCEV(const SCEV *, Value *, ScalarEvolution &);

int ReplaceValueWithConstant(Value *, unordered_map<Value *, int> &, TranslateStatus &);

void ReplaceSubstringWith(string &, string, string);
void SwitchToDOTRepresentation(string &base);
bool isConstantString(string);

#endif // LLVM_PLUSSUTILS_H
