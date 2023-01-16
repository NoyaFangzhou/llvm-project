//
// Created by noya-fangzhou on 11/5/21.
//

#include "BranchAnalysis.h"


#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/Statistic.h"
#include "InductionVarAnalysis.h"
#define DEBUG_TYPE 	"branch-analysis"

using namespace std;
using namespace llvm;

STATISTIC(BranchCount, "Number of branches in the function");
STATISTIC(PlussSupportBranchCount, "Number of branches that are suppported"
                                   "by Pluss analysis");

namespace BranchAnalysis {

ICmpInst *BranchAnalysis::GetLoopBranch(Loop *L, bool &isNegativeTransformed) {
  // TODO:
  //  currently, we assume the loop condition branch is always at the loop
  //  header. other passes in llvm assume there is a case that the condition
  //  branch is in the loop latch, but we find it not the case in our test cases.
  //  Using L->isCanonical() could be a better choice.
  BasicBlock *BlockToCheck = L->getHeader();
#if 0
  if (L->isLoopSimplifyForm()) {
    BlockToCheck = L->getHeader();
  } else {
    BlockToCheck = L->getLoopLatch();
  }
#endif
  LLVM_DEBUG(dbgs() << "Check block " << BlockToCheck->getName()
                    << " for branch\n");
  if (BranchInst *BI =
          dyn_cast_or_null<BranchInst>(BlockToCheck->getTerminator())) {
    if (BI->isConditional()) {
      LLVM_DEBUG(
          dbgs() << "Find condition " << *(BI->getCondition()) << "\n";
          for (unsigned i = 0; i < BI->getNumSuccessors(); i++) {
              dbgs() << BI->getSuccessor(i)->getName() << "\n";
          }
          );
      // BI->getSuccessor(0) BasicBlock to go if-true
      // BI->getSuccessor(1) BasicBlock to go if-false
      if (find(L->getBlocksVector().begin(), L->getBlocksVector().end(), BI->getSuccessor(0))
          != L->getBlocksVector().end()) {
        LLVM_DEBUG(dbgs() << "Will go to the loop body if the loop branch is true\n");
        isNegativeTransformed = false;
      } else {
        LLVM_DEBUG(dbgs() << "Will jump out the loop body if the loop branch is true\n");
        isNegativeTransformed = true;
      }
      return dyn_cast<ICmpInst>(BI->getCondition());
    }
  }
  return nullptr;
}

void BranchAnalysis::GetImmediateBranchInsideLoop(Loop *L,
                                             SmallVectorImpl<BranchInst *> &Branches)
{
  unordered_set<BasicBlock *> SubLBlocks;
  getBasicBlockInAllSubLoops(L, SubLBlocks, true);
  auto blockIter = L->block_begin();
  while (blockIter != L->block_end()) {
    BasicBlock *BlockToCheck = *blockIter;
    if (BlockToCheck == L->getHeader() || BlockToCheck == L->getLoopLatch()) {
      blockIter++;
      continue;
    }
    if (SubLBlocks.find(BlockToCheck) == SubLBlocks.end()) {
      BranchInst *BI = GetBranchInsideBlock(BlockToCheck);
      if (BI)
        Branches.push_back(BI);
    }
    blockIter++;
  }
}

BranchInst *BranchAnalysis::GetBranchInsideBlock(BasicBlock *BB)
{
  Instruction *Terminator = BB->getTerminator();
  if (BranchInst *BI = dyn_cast_or_null<BranchInst>(Terminator))  {
    if (BI->isConditional()) {
      return BI;
    }
  }
  return nullptr;
}

bool BranchAnalysis::IsPlussSupportBranch(Value *Cmp)
{
  if (!isa<CmpInst>(Cmp))
    return false;
  CmpInst *BranchCond = dyn_cast<CmpInst>(Cmp);
  bool ret = true;
  // check if all operand in this cmp is loop induction variable
  for (unsigned i = 0; i < BranchCond->getNumOperands(); i++) {
    Value *Op = BranchCond->getOperand(i);
    // check this Op is an arithmetic operation that contains the loop induction
    // variable only, f(i, j, k ...)
    if (!IsPlussSupportBranchImpl(Op)) {
      ret = false;
      break;
    }
  }
  return ret;
}

bool BranchAnalysis::IsPlussSupportBranchImpl(Value *V)
{
  if (!V)
    return false;
  if (isa<Constant>(V))
    return true;
  if (isa<Argument>(V))
    return true;
  Instruction *I = dyn_cast<Instruction>(V);
  LLVM_DEBUG(dbgs() << *I << " is pluss support?\n");
  if (!I)
    return false;
  bool ret = true;
  switch(I->getOpcode()) {
  case Instruction::FAdd:
  case Instruction::Add:
  case Instruction::FSub:
  case Instruction::Sub:
  case Instruction::FMul:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::LShr:
  case Instruction::Xor:
  case Instruction::And:
  case Instruction::Shl:
  case Instruction::Or: {
    for (unsigned i = 0; i < I->getNumOperands(); i++) {
      if (!IsPlussSupportBranchImpl(I->getOperand(i))) {
        ret = false;
        break;
      }
    }
    break;
  }
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::SExt:
  case Instruction::ZExt: {
    ret = IsPlussSupportBranchImpl(I->getOperand(0));
    break;
  }
  case Instruction::PHI: {
    PHINode *Phi = dyn_cast<PHINode>(I);
    ret = (InductionVarNameTable.find(Phi) != InductionVarNameTable.end());
    break;
  }
  case Instruction::Load:
  case Instruction::Alloca: {
    ret = false;
    break;
  }
  default:
    ret = false;
    break;
  }
  return ret;
}

char BranchAnalysisWrapperPass::ID = 0;
static RegisterPass<BranchAnalysisWrapperPass> X("branch-analysis",
                                      "Pass that analyzes basicblock"
                                      "inside a loop and finds those BasicBlocks"
                                      "inside a branch");

BranchAnalysisWrapperPass::BranchAnalysisWrapperPass() : FunctionPass(ID) {}

// BasicBlocks inside a branch
// A
void BranchAnalysisWrapperPass::FindBranchBasicBlockInLoop(Loop *L)
{

  SmallVector<BranchInst *, 8> Branches;
  BranchAnalyzer->GetImmediateBranchInsideLoop(L, Branches);
  BasicBlock *Header = L->getHeader(), *Latch = L->getLoopLatch();
  BranchCount += Branches.size();
  for (auto Branch : Branches) {
    if (BranchAnalyzer->IsPlussSupportBranch(Branch->getCondition()))
      PlussSupportBranchCount++;
    BasicBlock *BranchBlock = Branch->getParent();
    LLVM_DEBUG(dbgs() << "Block " << BranchBlock->getName() << " has Branch "
                      << *Branch << "\n");
    BasicBlock *IPostDom = ImmediatePostDominator(*PDT, BranchBlock);
    if (IPostDom) {
      LLVM_DEBUG(dbgs() << "The merge block of the condition "
                           " would be "
                        << IPostDom->getName() << "\n");
      // Now we find all pathes from BranchBlock to IPostDom
      SmallVector<Path, 2> Pathes;
      FindAllPathesBetweenTwoBlock(BranchBlock, IPostDom,
                                   Pathes);
      LLVM_DEBUG(
          for (auto path : Pathes) {
            for (auto BB : path) {
              dbgs() << BB->getName() << " -> ";
            }
            dbgs() << "\n";
          }
      );
    }
  }
}

void BranchAnalysisWrapperPass::DumpBranchBasicBlock()
{
//  for (auto BranchPair : BranchBlocks) {
//    dbgs() << BranchPair.first->getName() << " " << *(BranchPair.second);
//  }
}

/// Call for every function
bool BranchAnalysisWrapperPass::runOnFunction(Function &F)
{
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
  PDT = &(getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree());
  this->BranchAnalyzer = new BranchAnalysis(
      &getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
      getAnalysis<InductionVarAnalysis::InductionVarAnalysis>()
          .InductionVarNameTable);
  LoopInfo *LI = &(getAnalysis<LoopInfoWrapperPass>().getLoopInfo());
  auto toploopIter = LI->getTopLevelLoops().rbegin();
  while (toploopIter != LI->getTopLevelLoops().rend()) {
    Loop *L = *toploopIter;
    stack<Loop *> LoopNests;
    LoopNests.push(L);
    while (!LoopNests.empty()) {
      Loop *Head = LoopNests.top();
      LoopNests.pop();
      FindBranchBasicBlockInLoop(Head);
      for (auto SubL : Head->getSubLoops()) {
        LoopNests.push(SubL);
      }
    }
    toploopIter++;
  }
  //  F.viewCFG();
  LLVM_DEBUG(DumpBranchBasicBlock(););
  return false;
}

void BranchAnalysisWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const
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
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.addPreserved<PostDominatorTreeWrapperPass>();
  AU.addRequired<InductionVarAnalysis::InductionVarAnalysis>();
  AU.addPreserved<InductionVarAnalysis::InductionVarAnalysis>();
}


} // end of BranchAnalysis namespace
