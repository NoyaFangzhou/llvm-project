//
// Created by noya-fangzhou on 11/8/21.
//

#include "llvm/Transforms/Utils.h"
#include "llvm/ADT/Statistic.h"
#include "InductionVarAnalysis.h"

#define DEBUG_TYPE "indvar-analysis"

cl::opt<string> FunctionName("func", cl::init(string()), cl::NotHidden,
                             cl::desc("The function name to analyze"));

STATISTIC(InductionVarCount, "The number of loop induction variables"
                             "found inside the function");
STATISTIC(NumFunctionSkipped, "Number of functions skipped");

namespace InductionVarAnalysis {

char InductionVarAnalysis::ID = 0;
static RegisterPass<InductionVarAnalysis>
    X("indvar-analysis", "Pass that analyze the loop induction variable"
                      "and assign a name");

InductionVarAnalysis::InductionVarAnalysis() : FunctionPass(ID) {}

/// The name of each loop induction variable follows the rule:
/// c{loop_level}
/// i.e.
/// for (i = 0; i < N; i++) // id = 0, level = 0
///   for (j = 0; j < 1; j++) // id = 0, level = 1
///     ...
/// for (k = 0; k < N; k++) // id = 1, level = 0
///   ...
/// After naming
/// i: c0
/// j: c1
/// k: c0
/// i belongs to the first top level loop and it is in the top layer of the
/// loop nest, hence c0.
/// Both i and k are the induction variable of the top level loops, they share
/// the same name c0.
void InductionVarAnalysis::GenerateInductionVarNameForLoop(
    Loop *L, unsigned level)
{
  string name = "c" + to_string(level);
  PHINode *InductionPhi = L->getInductionVariable(SEWP->getSE());
  if (!InductionPhi)
    InductionPhi = L->getCanonicalInductionVariable();
  if (!InductionPhi)
    InductionPhi = getInductionVariable(L, SEWP->getSE());
  if (!InductionPhi)
    InductionPhi = getInductionVariableV2(L, SEWP->getSE());
  if (InductionPhi) {
    this->InductionVarNameTable[InductionPhi] = name;
  }
  for (auto SubL : L->getSubLoops()) {
    GenerateInductionVarNameForLoop(SubL, level+1);
  }
}

void InductionVarAnalysis::PrintInductionVarNameTable()
{
  if (this->InductionVarNameTable.empty()) {
    dbgs() << "No loop induction variable found\n";
    return;
  }
  dbgs() << this->InductionVarNameTable.size() << " InductionVar in table\n";
  for (auto TableElement : this->InductionVarNameTable) {
    dbgs() << *(TableElement.first) << " \t " << TableElement.second << "\n";
  }
}

/// Call for every function
bool InductionVarAnalysis::runOnFunction(Function &F)
{
  if (!FunctionName.empty() && F.getName().str().find(FunctionName) == string::npos) {
    // the function name is given, only the given function will be analyzed
    NumFunctionSkipped++;
    LLVM_DEBUG(dbgs() << F.getName() << " will be skipped\n");
    return false;
  }
#if defined(__APPLE__) && defined(__MACH__)
  LLVM_DEBUG(dbgs() << __FILE_NAME__ << " on " << F.getName() << "\n");
#elif defined(__linux__)
  LLVM_DEBUG(dbgs() << __FILE__ << " on " << F.getName() << "\n");
#endif
  LIWP = &getAnalysis<LoopInfoWrapperPass>();
  SEWP = &getAnalysis<ScalarEvolutionWrapperPass>();
  LoopInfo *LI = &(LIWP->getLoopInfo());
  auto toploopIter = LI->getTopLevelLoops().rbegin();
  while (toploopIter != LI->getTopLevelLoops().rend()) {
    GenerateInductionVarNameForLoop(*toploopIter, 0);
    toploopIter++;
  }
  InductionVarCount = this->InductionVarNameTable.size();
  LLVM_DEBUG(PrintInductionVarNameTable());
  return false;
}

void InductionVarAnalysis::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.setPreservesAll();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addRequiredID(LoopSimplifyID);
  AU.addPreservedID(LoopSimplifyID);
  AU.addRequiredID(LCSSAID);
  AU.addPreservedID(LCSSAID);
}


} // end of InductionVarAnalysis namespace

