//
// Created by noya-fangzhou on 4/20/22.
//

#include "ReferenceAnalysis.h"
#include "AccessGraphAnalysis.h"
#include "InductionVarAnalysis.h"
#include "LoopAnalysisWrapperPass.h"
#include "PlussAbstractionTreeAnalysis.h"
#include "ModelValidationAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Transforms/Utils.h"
#include <algorithm>

#define DEBUG_TYPE "ref-analysis"

namespace ReferenceAnalysis {

DependenceInfo *DI = nullptr;

char ReferenceAnalysis::ID = 0;
static RegisterPass<ReferenceAnalysis>
    X("ref-analysis", "Pass that analyze and group references");

ReferenceAnalysis::ReferenceAnalysis() : FunctionPass(ID) {}

void ReferenceAnalysis::ReferenceDependenceAnalysis(LoopTNode *LN)
{
  LLVM_DEBUG(dbgs() << "Dependence Analysis on Loop " << LN->getLoopStringExpr() << "\n");
  vector<RefTNode *> References;
  QueryEngine->FindAllRefsInLoop(LN, References);
  unsigned i = 0, j = 0;
  LLVM_DEBUG(dbgs() << "Dependence Pair:\n");
  for (i = 0; i < References.size(); i++) {
    for (j = 0; j < References.size(); j++) {
      RefTNode *A = References[i], *B = References[j];
      if (QueryEngine->areAccessToSameArray(A, B)) {
        LLVM_DEBUG(dbgs() << "";
          dbgs() << "(" << A->getRefExprString() << " <=> " << B->getRefExprString()
               << ")\n";);
      }
    }
  }

  unsigned ParallelLoopLevel = LN->getLoopLevel();
  LLVM_DEBUG(dbgs() << "Level " << ParallelLoopLevel << "\n");

  for (i = 0; i < References.size(); i++) {
    for (j = 0; j < References.size(); j++) {
      RefTNode *A = References[i], *B = References[j];
      std::vector<char> Dep;
      if (QueryEngine->areAccessToSameArray(A, B)) {
        LLVM_DEBUG(
            dbgs() << "Dep[";
            dbgs() << A->getRefExprString() << "][" << B->getRefExprString() << "] = ";
            );
        // do dependence testing when A and B are the reference to the same
        // array. D is null means A and B has no dependence
        if (auto D = DI->depends(A->getMemOp(), B->getMemOp(), true)) {
          unsigned Levels = D->getLevels();
          char Direction;
          for (unsigned II = 1; II <= Levels; ++II) {
            if (II != ParallelLoopLevel+1)
              continue;
            const SCEV *Distance = D->getDistance(II);
            LLVM_DEBUG(
                if (Distance)
                    dbgs() << "Distance at level " << II << " is " << *Distance << "\n");
            const SCEVConstant *SCEVConst =
                dyn_cast_or_null<SCEVConstant>(Distance);
            if (SCEVConst) {
              const ConstantInt *CI = SCEVConst->getValue();
              if (CI->isNegative())
                Direction = '<';
              else if (CI->isZero())
                Direction = '=';
              else
                Direction = '>';
              Dep.push_back(Direction);
            } else if (D->isScalar(II)) {
              Direction = 'S';
              Dep.push_back(Direction);
            } else {
              unsigned Dir = D->getDirection(II);
              if (Dir == Dependence::DVEntry::LT ||
                  Dir == Dependence::DVEntry::LE)
                Direction = '<';
              else if (Dir == Dependence::DVEntry::GT ||
                       Dir == Dependence::DVEntry::GE)
                Direction = '>';
              else if (Dir == Dependence::DVEntry::EQ)
                Direction = '=';
              else
                Direction = '*';
              Dep.push_back(Direction);
            }
          }
          LLVM_DEBUG(
              dbgs() << "(";
              for (unsigned i = 0; i < Dep.size(); i++) {
                  dbgs() << Dep[i];
                  if (i != Dep.size()-1)
                    dbgs() << ", ";
              }
              dbgs() << ")\n";
              );
        } else {
          LLVM_DEBUG(dbgs() << "(-)\n" );
        }
      }
    }
  }

}

unsigned ReferenceAnalysis::GetNumberOfDependenceReferences(RefTNode *Target)
{
  unsigned count = 0;
  if (LoopTNode *ParentLoop = QueryEngine->GetImmdiateLoopDominator(Target)) {
    vector<RefTNode *> References;
    QueryEngine->FindAllRefsInLoop(ParentLoop, References);
    for (auto ref : References) {
      if (QueryEngine->areAccessToSameArray(ref, Target))
        count += 1;
    }
  }
  return count;
}


bool ReferenceAnalysis::runOnFunction(Function &F) {
  if (!FunctionName.empty() && F.getName().str().find(FunctionName) == string::npos) {
    // the function name is given, only the given function will be analyzed
    LLVM_DEBUG(dbgs() << F.getName() << " will be skipped\n");
    return false;
  }
#if defined(__APPLE__) && defined(__MACH__)
  LLVM_DEBUG(dbgs() << __FILE_NAME__ << " on " << F.getName() << "\n");
#elif defined(__linux__)
  LLVM_DEBUG(dbgs() << __FILE__ << " on " << F.getName() << "\n");
#endif
  SEWP = &getAnalysis<ScalarEvolutionWrapperPass>();
  BAWP = &getAnalysis<BranchAnalysis::BranchAnalysisWrapperPass>();
  DI = &getAnalysis<DependenceAnalysisWrapperPass>().getDI();

  TreeRoot = getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>().TreeRoot;

  /* utils */
  Translator = new StringTranslator(&SEWP->getSE(), getAnalysis<InductionVarAnalysis::InductionVarAnalysis>()
      .InductionVarNameTable);
  QueryEngine = &getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>()
      .getQueryEngine();

  auto topiter = TreeRoot->neighbors.begin();
  for (; topiter != TreeRoot->neighbors.end(); ++topiter) {
    if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(*topiter)) {
      ReferenceDependenceAnalysis(Loop);
    }
  }
  topiter = TreeRoot->neighbors.begin();
  vector<RefTNode *> References;
  unsigned i = 0;
  for (; topiter != TreeRoot->neighbors.end(); ++topiter) {
    if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(*topiter)) {
      QueryEngine->FindAllRefsInLoop(Loop, References);
      for (i = 0; i < References.size(); i++) {
        RefTNode *RefNode = References[i];
        PerLoopArraysMap[Loop].insert(RefNode->getArrayNameString());

        unsigned reuse_dep_refcnt = GetNumberOfDependenceReferences(RefNode);
        ReferenceDependenceCount[RefNode] = reuse_dep_refcnt;
      }
      References.clear();
    }
  }

  return false;
}

void ReferenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addRequired<DependenceAnalysisWrapperPass>();
  AU.addPreserved<DependenceAnalysisWrapperPass>();
  AU.addRequiredID(LoopSimplifyID);
  AU.addPreservedID(LoopSimplifyID);
  AU.addRequiredID(LCSSAID);
  AU.addPreservedID(LCSSAID);
  AU.addRequired<InductionVarAnalysis::InductionVarAnalysis>();
  AU.addRequired<BranchAnalysis::BranchAnalysisWrapperPass>();
  AU.addRequired<PlussLoopAnalysis::LoopAnalysisWrapperPass>();
  AU.addPreserved<PlussLoopAnalysis::LoopAnalysisWrapperPass>();
  AU.addRequired<TreeAnalysis::PlussAbstractionTreeAnalysis>();
  AU.addPreserved<TreeAnalysis::PlussAbstractionTreeAnalysis>();
}



} // end of namespace
