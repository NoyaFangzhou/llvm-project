//
//  LoopAnalysisWrapperPass.cpp
//  LLVM
//
//  Created by noya-fangzhou on 10/13/21.
//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "LoopAnalysisWrapperPass.h"
#include "InductionVarAnalysis.h"

#define DEBUG_TYPE 	"loop-analysis"

cl::opt<bool> ViewAbstractionTree(
    "view-tree", cl::init(false), cl::NotHidden,
    cl::desc("Output the abstraction tree in DOT format"));

cl::opt<bool> ViewInductionVariableDependency(
    "view-indvar-dep", cl::init(false), cl::NotHidden,
    cl::desc("Output the induction variable dependency tree in DOT format"));

STATISTIC(AccessNodeCount, "Number of Reference Node in the tree");
STATISTIC(SupportAccessNodeCount, "Number of Reference Node in the tree that "
                                  "is supported by PLUSS");
STATISTIC(LoopCount, "Number of Loops examined");
STATISTIC(LoopNotSupport, "Number of Loops that cannot be analyzed");
STATISTIC(PlussParallelLoopCount, "Number of parallel Loops ");
STATISTIC(MemoryOpCount, "Number of Memory instructions examined");
STATISTIC(MemoryOpNotSupport, "Number of Memory instructions"
                              " that cannot be analyzed");
STATISTIC(ParallelDescLoopCount, "Number of parallel Loops with "
                                 "negative stride ");
STATISTIC(ParallelAscLoopCount, "Number of parallel Loops with positive stride");
STATISTIC(BranchInversedCount, "Number of branches whose predicate is inversed "
                               "by the loop transformation");

using namespace std;
using namespace llvm;

namespace PlussLoopAnalysis {

char LoopAnalysisWrapperPass::ID = 0;
static RegisterPass<LoopAnalysisWrapperPass> X("loop-analysis",
                                    "Pass that analyze the loop structure and "
                                    "build the abstract loop tree");

LoopAnalysisWrapperPass::LoopAnalysisWrapperPass() : FunctionPass(ID) {}


Instruction* IsInterestingMemOp(Value *V)
{
  if (!V)
    return nullptr;
  Instruction *MemOpInst = dyn_cast<Instruction>(V);
  if (!MemOpInst)
    return nullptr;
  unsigned operand_to_check = 0;
  if (isa<StoreInst>(MemOpInst))
    operand_to_check = 1;
  if (isa<GetElementPtrInst>(MemOpInst->getOperand(operand_to_check))) {
    return MemOpInst;
  } else if (MemOpInst->getOperand(operand_to_check)->getType()->isPointerTy() &&
      MemOpInst->getOperand(operand_to_check)->hasName()) {
    // This tries to handle a special case.
    // When access an array at index 0, instead of generating
    // a getelementptr, the load instruction loads the pointer directly.
    return MemOpInst;
  } else if (MemOpInst->getOperand(operand_to_check)->getType()->isPointerTy() &&
             isa<BitCastInst>(MemOpInst->getOperand(operand_to_check))) {
    // array type conversion
    BitCastInst *BitCast = dyn_cast<BitCastInst>(MemOpInst->getOperand(operand_to_check));
    return IsInterestingMemOp(BitCast);
  }
  LLVM_DEBUG(dbgs() << "MemOp " << *MemOpInst << " is not interesting\n");
  return nullptr;
}


void FindMemOpsInBasicBlock(BasicBlock *BB, vector<Instruction *> &MemOps)
{
   LLVM_DEBUG(dbgs() << "Traverse Instructions in " << BB->getName() << " \n");
  unordered_set<Instruction *> VisitedGEPInsts;
  for (auto II = BB->begin(); II != BB->end(); ++II) {
    if (isa<LoadInst>(II) && !II->getType()->isPointerTy()) {
      MemoryOpCount++;
      Instruction *ReadInst = &(*II);
      if (Instruction *MemReadInst = IsInterestingMemOp(ReadInst)) {
        MemOps.push_back(MemReadInst);
      }
#if 0
      if (isa<GetElementPtrInst>(MemReadInst->getOperand(0))) {
        MemOps.push_back(MemReadInst);
      } else if (MemReadInst->getOperand(0)->getType()->isPointerTy() &&
               MemReadInst->getOperand(0)->hasName()) {
        // This tries to handle a special case.
        // When access an array at index 0, instead of generating
        // a getelementptr, the load instruction loads the pointer directly.
        MemOps.push_back(MemReadInst);
      }
#endif
    } else if (isa<StoreInst>(II) && !II->getType()->isPointerTy()) {
      MemoryOpCount++;
      Instruction *WriteInst = &(*II);
      if (Instruction *MemWriteInst = IsInterestingMemOp(WriteInst)) {
        MemOps.push_back(MemWriteInst);
      }
#if 0
      if (isa<GetElementPtrInst>(MemWriteInst->getOperand(1))) {
        MemOps.push_back(MemWriteInst);
      } else if (MemWriteInst->getOperand(1)->getType()->isPointerTy() &&
          MemWriteInst->getOperand(1)->hasName()) {
        // This tries to handle a special case.
        // When access an array at index 0, instead of generating
        // a getelementptr, the load instruction loads the pointer directly.
        MemOps.push_back(MemWriteInst);
      }
#endif
    }
  }
}

void LoopAnalysisWrapperPass::AddRefNodeToList(BasicBlock *Block,
                            vector<SPSTNode *> &SPSNodeList) {
  vector<Instruction *> MemOpInsts;
  FindMemOpsInBasicBlock(Block, MemOpInsts);
  for (auto MemOpInst : MemOpInsts) {
//    LLVM_DEBUG(dbgs() << "ADD " << *MemOpInst << "\n");
    if (isa<LoadInst>(MemOpInst)) {
      LoadInst *MemReadInst = cast<LoadInst>(MemOpInst);
      GetElementPtrInst *GEPInst =
          dyn_cast<GetElementPtrInst>(MemReadInst->getOperand(0));
      if (GEPInst) {
        RefTNode *RefNode =
            new RefTNode(MemReadInst, GEPInst, this->SEWP->getSE());
        if (RefNode->getBase()) {
          SupportAccessNodeCount++;
          GenerateRefExpr(RefNode);
        }
        LLVM_DEBUG(dbgs() << "ADD " << *MemOpInst << RefNode->getRefExprString() << "\n");
        AccessToNodeMapping[MemReadInst] = RefNode;
        SPSNodeList.push_back(RefNode);
      } else if (MemReadInst->getOperand(0)->getType()->isPointerTy()) {
        // handle the case of loading a pointer directly, which means the first
        // element of an array
        RefTNode *RefNode =
            new RefTNode(MemReadInst, nullptr, this->SEWP->getSE());
        if (RefNode->getBase()) {
          SupportAccessNodeCount++;
          GenerateRefExpr(RefNode);
        }
        LLVM_DEBUG(dbgs() << "ADD " << *MemOpInst << RefNode->getRefExprString() << "\n");
        AccessToNodeMapping[MemReadInst] = RefNode;
        SPSNodeList.push_back(RefNode);
      }
    } else if (isa<StoreInst>(MemOpInst)) {
      StoreInst *MemWriteInst = cast<StoreInst>(MemOpInst);
      GetElementPtrInst *GEPInst =
          dyn_cast<GetElementPtrInst>(MemWriteInst->getOperand(1));
      if (GEPInst) {
        RefTNode *RefNode =
            new RefTNode(MemWriteInst, GEPInst, this->SEWP->getSE());
        if (RefNode->getBase()) {
          SupportAccessNodeCount++;
          GenerateRefExpr(RefNode);
        }
        LLVM_DEBUG(dbgs() << "ADD " << *MemOpInst << RefNode->getRefExprString()
                          << "\n");
        AccessToNodeMapping[MemWriteInst] = RefNode;
        SPSNodeList.push_back(RefNode);
      }  else if (MemWriteInst->getOperand(1)->getType()->isPointerTy()) {
        // handle the case of writing a pointer directly, which means the first
        // element of an array
        RefTNode *RefNode =
            new RefTNode(MemWriteInst, nullptr, this->SEWP->getSE());
        if (RefNode->getBase()) {
          SupportAccessNodeCount++;
          GenerateRefExpr(RefNode);
        }
        LLVM_DEBUG(dbgs() << "ADD " << *MemOpInst << RefNode->getRefExprString() << "\n");
        AccessToNodeMapping[MemWriteInst] = RefNode;
        SPSNodeList.push_back(RefNode);
      }
    } else if (isa<BitCastInst>(MemOpInst)) {
      BitCastInst *MemCastInst = cast<BitCastInst>(MemOpInst);
      GetElementPtrInst *GEPInst =
          dyn_cast<GetElementPtrInst>(MemCastInst->getOperand(0));
      RefTNode *RefNode = new RefTNode(MemCastInst, GEPInst, this->SEWP->getSE());
      if (RefNode->getBase()) {
        SupportAccessNodeCount++;
        GenerateRefExpr(RefNode);
      }
      LLVM_DEBUG(dbgs() << "ADD " << *MemOpInst << RefNode->getRefExprString() << "\n");
      AccessToNodeMapping[MemCastInst] = RefNode;
      SPSNodeList.push_back(RefNode);
    }
  }
  if (!SPSNodeList.empty())
    LLVM_DEBUG(dbgs() << "Find " << SPSNodeList.size() << " accesses\n");
}

// The idea is we first build trees for all loops, then decorate refnode and
// branch node on this tree.
SPSTNode *LoopAnalysisWrapperPass::BuildTreeForLoopImpl(Loop *L,
                                                        unordered_set<BasicBlock *> &Visited,
                                                        unsigned LoopLevel)
{
  bool IsParallel = IsPlussParallelLoop(L);
  LoopTNode *LoopNode = new LoopTNode(L, LoopLevel, LoopCnt, IsParallel,
                                      this->SEWP->getSE());
  LoopCnt ++;
  LoopCount ++;
  if (IsParallel) { PlussParallelLoopCount++; }
  GenerateLoopExpr(LoopNode);
  LoopToNodeMapping[L] = LoopNode;
  LLVM_DEBUG(dbgs() << " -- Loop " << L->getHeader()->getName() << ", " << L->getLoopLatch()->getName() << " \n");
  auto subloop_riter = L->getSubLoops().rbegin();
  for (; subloop_riter != L->getSubLoops().rend(); ++subloop_riter) {
    LoopNode->addNeighbors(BuildTreeForLoopImpl(*subloop_riter,
                                                Visited, LoopLevel+1));
  }
  // mark all basicblocks in the loop visited
  for (auto Block : L->getBlocks()) {
    if (Visited.find(Block) == Visited.end())
      Visited.insert(Block);
  }
  // next we will add branch node into the tree
  return LoopNode;
}

void LoopAnalysisWrapperPass::AppendBranchNodesOnTreeImpl(SPSTNode *NodeInTree)
{
  if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(NodeInTree)) {
    // We next traverse all blocks inside the loop (exclude those belongs to its
    // subloops, and find all branches.
    Loop *LoopToDecorate = LoopNode->getLoop();
    unordered_set<BasicBlock *> SubLoopBlocks;
    getBasicBlockInAllSubLoops(LoopToDecorate, SubLoopBlocks, true);
    for (auto Block : LoopToDecorate->getBlocksVector()) {
      if (Block == LoopToDecorate->getHeader() ||
          Block == LoopToDecorate->getLoopLatch() ||
          SubLoopBlocks.find(Block) != SubLoopBlocks.end()) {
        continue;
      }
      LLVM_DEBUG(dbgs() << "Block " << Block->getName() << "  \n");
      BranchInst *BI = BAWP->getBranchAnalyzer().GetBranchInsideBlock(Block);
      if (BI &&
          BAWP->getBranchAnalyzer().IsPlussSupportBranch(BI->getCondition())) {
        LLVM_DEBUG(
            dbgs() << "-- Branch " << *BI << "\n";
            dbgs() << "-- True : " << BI->getSuccessor(0)->getName() << "\n";
            dbgs() << "-- False : " << BI->getSuccessor(1)->getName() << "\n";);
        SmallVector<Path, 16> Pathes;
        // It is possible that the immediate post dominator of one branch
        // could also have branches inside.
        BasicBlock *IPostDom = ImmediatePostDominator(
            getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree(),
            BI->getParent());
        if (!IPostDom) {
          LLVM_DEBUG(dbgs() << "No Immediate post dominator of this branch,"
                               "we ignor it\n");
          continue;
        }
        LLVM_DEBUG(dbgs() << "Branch merges at " << IPostDom->getName()
                          << "\n");
        // this block is not in any loops, we move all access nodes
        // inside down to the branch path.
        vector<Instruction *> MemOps;
        vector<RefTNode *> RefsInMergePoint;
        FindMemOpsInBasicBlock(IPostDom, MemOps);
        for (auto memop : MemOps) {
          if (AccessToNodeMapping.find(memop) == AccessToNodeMapping.end())
            continue;
          LLVM_DEBUG(dbgs()
                     << "Remove reference at Branch merge point "
                     << AccessToNodeMapping[memop]->getRefExprString()
                     << "from " << LoopNode->getLoopStringExpr() << "\n");
          LLVM_DEBUG(dbgs() << "Before " << LoopNode->getNumNeighbors() << " ");
          LoopNode->eraseFromNeighbor(AccessToNodeMapping[memop]);
          RefsInMergePoint.push_back(AccessToNodeMapping[memop]);
          LLVM_DEBUG(dbgs()
                     << "After " << LoopNode->getNumNeighbors() << " \n");
        }
        // we will traverse the blocks in CFG related to
        // this branch and build a BranchTNode.
        // During the traversal, blocks after the branch block will also be
        // traversed. To avoid duplicate traversal, we mark those blocks visited
        //
        // One special case is that, the IPostDom of one branch can also have
        // branch conditions inside.  In this case, it needs to be traversed
        // here
        BranchTNode *BranchNode = new BranchTNode(BI->getCondition());
        GenerateBranchExpr(BranchNode);
        BasicBlock *TrueBlock = BI->getSuccessor(0);
        BasicBlock *FalseBlock = BI->getSuccessor(1);
        // traverse these branches and move all those loopnode at the bottom
        // of each branch
        FindAllPathesBetweenTwoBlock(TrueBlock, IPostDom, Pathes);
        unordered_set<BasicBlock *> visited;
        for (auto path : Pathes) {
          for (auto block : path) {
            if (visited.find(block) == visited.end() && block != IPostDom) {
              visited.insert(block);
              vector<Loop *> SubLInBranch;
              FindSubLoopsContainBlock(LoopToDecorate, block, SubLInBranch);
              if (!SubLInBranch.empty()) {
                // this block belongs to a subloop, we move its loop node to the
                // false branch.
                // since all access node is attached below this loop node, they
                // are moved automatically
                for (auto subloop : SubLInBranch) {
                  LLVM_DEBUG(dbgs()
                             << "Remove subloop at true "
                             << LoopToNodeMapping[subloop]->getLoopStringExpr()
                             << "from " << LoopNode->getLoopStringExpr()
                             << "\n");
                  LLVM_DEBUG(dbgs() << "Before " << LoopNode->getNumNeighbors()
                                    << " ");
                  LoopNode->eraseFromNeighbor(LoopToNodeMapping[subloop]);
                  LLVM_DEBUG(dbgs() << "After " << LoopNode->getNumNeighbors()
                                    << " \n");
                  BranchNode->addTrueNode(LoopToNodeMapping[subloop]);
                }
              } else {
                // this block is not in any loops, we move all access nodes
                // inside down to the branch path.
                MemOps.clear();
                FindMemOpsInBasicBlock(block, MemOps);
                for (auto memop : MemOps) {
                  if (AccessToNodeMapping.find(memop) == AccessToNodeMapping.end())
                    continue;
                  LLVM_DEBUG(dbgs()
                                 << "Remove reference at true "
                                 << AccessToNodeMapping[memop]->getRefExprString()
                                 << "from " << LoopNode->getLoopStringExpr()
                                 << "\n");
                  LLVM_DEBUG(dbgs() << "Before " << LoopNode->getNumNeighbors()
                                    << " ");
                  LoopNode->eraseFromNeighbor(AccessToNodeMapping[memop]);
                  LLVM_DEBUG(dbgs() << "After " << LoopNode->getNumNeighbors()
                                    << " \n");
                  BranchNode->addTrueNode(AccessToNodeMapping[memop]);
                }
              }
            }
          }
        }
        Pathes.clear();
        visited.clear();
        FindAllPathesBetweenTwoBlock(FalseBlock, IPostDom, Pathes);
        for (auto path : Pathes) {
          for (auto block : path) {
            if (visited.find(block) == visited.end() && block != IPostDom) {
              visited.insert(block);
              vector<Loop *> SubLInBranch;
              FindSubLoopsContainBlock(LoopToDecorate, block, SubLInBranch);
              if (!SubLInBranch.empty()) {
                // this block belongs to a subloop, we move its loop node to the
                // false branch.
                // since all access node is attached below this loop node, they
                // are moved automatically
                for (auto subloop : SubLInBranch) {
                  LLVM_DEBUG(dbgs()
                                 << "Remove subloop at false "
                                 << LoopToNodeMapping[subloop]->getLoopStringExpr()
                                 << "from " << LoopNode->getLoopStringExpr() << "\n");
                  LLVM_DEBUG(dbgs()
                                 << "Before " << LoopNode->getNumNeighbors() << " ");
                  LoopNode->eraseFromNeighbor(LoopToNodeMapping[subloop]);
                  LLVM_DEBUG(dbgs()
                                 << "After " << LoopNode->getNumNeighbors() << " \n");
                  BranchNode->addFalseNode(LoopToNodeMapping[subloop]);
                }
              } else {
                // this block is not in any loops, we move all access nodes
                // inside down to the branch path.
                MemOps.clear();
                FindMemOpsInBasicBlock(block, MemOps);
                for (auto memop : MemOps) {
                  if (AccessToNodeMapping.find(memop) == AccessToNodeMapping.end())
                    continue;
                  LLVM_DEBUG(dbgs()
                             << "Remove reference at false "
                             << AccessToNodeMapping[memop]->getRefExprString()
                             << "from " << LoopNode->getLoopStringExpr()
                             << "\n");
                  LLVM_DEBUG(dbgs() << "Before " << LoopNode->getNumNeighbors()
                                    << " ");
                  LoopNode->eraseFromNeighbor(AccessToNodeMapping[memop]);
                  LLVM_DEBUG(dbgs() << "After " << LoopNode->getNumNeighbors()
                                    << " \n");
                  BranchNode->addFalseNode(AccessToNodeMapping[memop]);
                }
              }
            }
          }
        }
        LoopNode->addNeighbors(BranchNode);
        // we now append all those references at the merge point into the loop
        for (auto ref : RefsInMergePoint) {
          LoopNode->addNeighbors(ref);
        }
      }
    }
    // Append branch node to all its child
    for (auto neighbor_iter = LoopNode->neighbors.begin();
         neighbor_iter != LoopNode->neighbors.end(); neighbor_iter++) {
      AppendBranchNodesOnTreeImpl(*neighbor_iter);
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(NodeInTree)) {
    for (auto neighbor_iter = Dummy->neighbors.begin();
         neighbor_iter != Dummy->neighbors.end(); neighbor_iter++) {
      AppendBranchNodesOnTreeImpl(*neighbor_iter);
    }
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(NodeInTree)) {
    AppendBranchNodesOnTreeImpl(Branch->neighbors[0]);
    AppendBranchNodesOnTreeImpl(Branch->neighbors[1]);
  } else if (RefTNode *RefNode = dynamic_cast<RefTNode *>(NodeInTree)) {
    // do nothing
  }
}

void LoopAnalysisWrapperPass::AppendRefNodesOnTreeImpl(SPSTNode *NodeInTree)
{
  if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(NodeInTree)) {
    // We next traverse all blocks inside the loop (exclude those belongs to its
    // subloops, and find all reference nodes.
    vector<SPSTNode *> newNeighborList;
    Loop *LoopToDecorate = LoopNode->getLoop();
    unordered_set<BasicBlock *> SubLoopBlocks;
    getBasicBlockInAllSubLoops(LoopToDecorate, SubLoopBlocks, true);
    for (auto Block : LoopToDecorate->getBlocksVector()) {
      if (SubLoopBlocks.find(Block) != SubLoopBlocks.end()) {
        Loop *SubL = FindSubloopContainsBlockInLoop(LoopToDecorate, Block);
        // avoid dumplication, one subloop can contain multiple basicblocks.
        if (find(newNeighborList.begin(), newNeighborList.end(),
                 LoopToNodeMapping[SubL]) == newNeighborList.end()) {
          newNeighborList.push_back(LoopToNodeMapping[SubL]);
        }
        continue;
      }
      LLVM_DEBUG(dbgs() << "Block " << Block->getName() << "  \n");
      vector<SPSTNode *> RefNodes;
      AddRefNodeToList(Block, RefNodes);
      for (auto Ref : RefNodes) {
        newNeighborList.push_back(Ref);
      }
      AccessNodeCount += RefNodes.size();
    }
    // Append branch node to all its child
    for (auto neighbor_iter = LoopNode->neighbors.begin();
         neighbor_iter != LoopNode->neighbors.end(); neighbor_iter++) {
      AppendRefNodesOnTreeImpl(*neighbor_iter);
    }
    LoopNode->neighbors.clear();
    for (auto node : newNeighborList) {
      LoopNode->neighbors.push_back(node);
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(NodeInTree)) {
    for (auto neighbor_iter = Dummy->neighbors.begin();
         neighbor_iter != Dummy->neighbors.end(); neighbor_iter++) {
      AppendRefNodesOnTreeImpl(*neighbor_iter);
    }
  }
//  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(NodeInTree)) {
//    AppendRefNodesOnTreeImpl(Branch->neighbors[0]);
//    AppendRefNodesOnTreeImpl(Branch->neighbors[1]);
//  }
}

/*
// Create a Node for L,
// The build phase starts from the innermost loop of L
// The inner loop contains branches and references only
SPSTNode *LoopAnalysisWrapperPass::BuildTreeForLoopImpl(Loop *L,
                                            unordered_set<BasicBlock *> &Visited,
                                            unsigned LoopLevel)
{
vector<pair<Loop *, unsigned>> LoopNest;
unsigned level = 0;
stack<pair<Loop *, unsigned>> s;
s.push(make_pair(L, 0));
while (!s.empty()) {
  Loop *Top = s.top().first;
  LoopNest.push_back(s.top());
  level = s.top().second;
  s.pop();
  for (auto SubL : Top->getSubLoops()) {
    s.push(make_pair(SubL, level+1));
  }
}



// all loops will be traversed in inner-to-outer order
// adjacent loops will be traversed in order.

// we group those LoopTNodes with their corresponding level in the tree
unordered_map<Loop *, LoopTNode *> LoopNodeMapping;
unordered_set<BasicBlock *> IPostDomSet;
unordered_set<BasicBlock *> BlockToSkip;
auto looprIter = LoopNest.rbegin();
while (looprIter != LoopNest.rend()) {
  Loop *LoopToBuild = looprIter->first;
  LLVM_DEBUG(
      dbgs() << "[" << looprIter->second << "] loop "
             << LoopToBuild->getHeader()->getName() << " \n");

  LLVM_DEBUG(dbgs() << "Build tree for loop "
                    << LoopToBuild->getHeader()->getName() << "\n";
             auto blockIter = LoopToBuild->block_begin();
             while (blockIter != LoopToBuild->block_end()) {
               dbgs() << (*blockIter)->getName() << " -> ";
               blockIter++;
             }
             dbgs() << "\n";
  );
  unsigned current_level = looprIter->second;
  bool IsParallel = IsPlussParallelLoop(L);
  LoopTNode *LoopNode = new LoopTNode(LoopToBuild, current_level, IsParallel,
                                      this->SEWP->getSE());
  if (IsParallel) { PlussParallelLoopCount++; }
  GenerateLoopExpr(LoopNode);
  auto blockIter = LoopToBuild->block_begin();
  // When iterating blocks inside the LoopToBuild, the iterator does not
  // guarantee that blocks will be traversed in order.
  while (blockIter != LoopToBuild->block_end()) {
    BasicBlock *Block = *blockIter;
    if (BlockToSkip.find(Block) != BlockToSkip.end()) {
      LLVM_DEBUG(dbgs() << "SKIP " << Block->getName() << " \n");
      blockIter++;
      continue;
    } else if (Visited.find(Block) != Visited.end()) {
      //
      // 1) The loop has no subloops and this block is visited because of
      // branch traversal
      // 2) The loop has no subloops and this block is visited because of
      // branch traversal. This is a special case where the immediate post
      // dominator of one branch could also have branches inside, hence needs
      // to be traversed twice.
      // 3) The loop has subloops and this block belongs to one of its subloop
      //
      // if case 1, we do nothing and continue;
      // if case 3, we append the loopnode of this subloop and we jump
      // all other blocks inside this subloop
      if (!LoopToBuild->getSubLoopsVector().empty()) {
        Loop *SubL = FindSubloopContainsBlockInLoop(LoopToBuild, Block);
        if (SubL) {
          LLVM_DEBUG(dbgs()
                     << "CASE 2: Block " << Block->getName() << " is in "
                     << "Subloop" << SubL->getHeader()->getName() << "\n");
          // we do not have to handle the case where SubL not in LoopNodeMapping
          // all loops are traversed in inner-to-order order, hence subloop
          // will always be traversed before their parents.
          LoopNode->addNeighbors(LoopNodeMapping[SubL]);
          // we should skip all blocks in SubL
          for (auto SubLBlock : SubL->getBlocks()) {
            BlockToSkip.insert(SubLBlock);
          }
        }
        blockIter++;
        continue;
      } else if (IPostDomSet.find(Block) != IPostDomSet.end()) {
        // here
      } else {
        blockIter++;
        continue;
      }
    }
    Visited.insert(Block);
    vector<SPSTNode *> RefNodes;
    AddRefNodeToList(Block, RefNodes);
    vector<SPSTNode *>::iterator refIter = RefNodes.begin();
    while (refIter != RefNodes.end()) {
      LoopNode->addNeighbors(*refIter);
      refIter++;
    }
    AccessNodeCount += RefNodes.size();
    if (Block == LoopToBuild->getHeader()
        || Block == LoopToBuild->getLoopLatch()) {
      blockIter++;
      continue;
    }
    BranchInst *BI =
        BAWP->getBranchAnalyzer().GetBranchInsideBlock(Block);
    if (BI &&
        BAWP->getBranchAnalyzer().IsPlussSupportBranch(BI->getCondition())) {
      LLVM_DEBUG(
          dbgs() << "-- Branch " << *BI << "\n";
          dbgs() << "-- True : " << BI->getSuccessor(0)->getName() << "\n";
          dbgs() << "-- False : " << BI->getSuccessor(1)->getName() << "\n";);
      SmallVector<Path, 16> Pathes;
      BasicBlock *IPostDom = ImmediatePostDominator(
          getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree(),
          BI->getParent());
      if (!IPostDom) {
        LLVM_DEBUG(dbgs() << "No Immediate post dominator of this branch,"
                             "we ignor it\n");
        continue;
      }
      IPostDomSet.insert(IPostDom);
      // we will traverse the blocks in CFG related to
      // this branch and build a BranchTNode.
      // During the traversal, blocks after the branch block will also be
      // traversed. To avoid duplicate traversal, we mark those blocks visited
      //
      // One special case is that, the IPostDom of one branch can also have
      // branch conditions inside.  In this case, it needs to be traversed
      // here
      FindAllPathesBetweenTwoBlock(Block, IPostDom, Pathes);
      for (auto path : Pathes) {
        for (auto block : path) {
          if (Visited.find(block) == Visited.end())
            Visited.insert(block);
        }
      }
      SPSTNode *BrNode = BuildBranchNodeFor(BI);
      LoopNode->addNeighbors(BrNode);
    }
    blockIter++; // move to the next block in LoopToBuild
  }
  // move to the deepest loop of its next sibling loop (curr_level >= next_level)
  // or its parent loop (curr_level < next_level)
  LoopNodeMapping[looprIter->first] = LoopNode;
  looprIter++;
}
return LoopNodeMapping[L];
}

SPSTNode * LoopAnalysisWrapperPass::BuildTreeForLoopImpl(Loop *L,
                                           unordered_set<BasicBlock *> &Visited,
                                           unsigned LoopLevel)
{
LLVM_DEBUG(dbgs() << "Build tree for loop "
                  << L->getHeader()->getName() << "\n");
bool IsParallel = IsPlussParallelLoop(L);
LoopTNode *LoopNode = new LoopTNode(L, LoopLevel, IsParallel,
                                    this->SEWP->getSE());
if (IsParallel) {
  PlussParallelLoopCount++;
}
GenerateLoopExpr(LoopNode);
auto blockIter = L->block_begin();
auto subloopIter = L->getSubLoops().begin();
while (blockIter != L->block_end()) {
  if (!L->getSubLoops().empty() && subloopIter != L->getSubLoops().end() &&
      (*subloopIter)->contains(*blockIter)) {
    Loop *SubL = *subloopIter;
    LoopNode->addNeighbors(BuildTreeForLoopImpl(SubL, Visited, LoopLevel+1));
    while (SubL->contains(*blockIter)) {
      blockIter++;
    }
    subloopIter++; // we move to the next subloop
  } else if (Visited.find(*blockIter) != Visited.end()) {
    blockIter ++;
  } else {
    Visited.insert(*blockIter);
    vector<SPSTNode *> RefNodes;
    AddRefNodeToList(*blockIter, RefNodes);
    vector<SPSTNode *>::iterator  refIter = RefNodes.begin();
    while (refIter != RefNodes.end()) {
      LoopNode->addNeighbors(*refIter);
      refIter++;
    }
    AccessNodeCount += RefNodes.size();
    BranchInst *BI = BAWP->getBranchAnalyzer().GetBranchInsideBlock(*blockIter);
    if (BI && BAWP->getBranchAnalyzer().IsPlussSupportBranch(BI->getCondition())) {
      BranchTNode *BrNode = new BranchTNode(BI->getCondition());
      LoopNode->addNeighbors(BrNode);
      LLVM_DEBUG(dbgs() << "if ->" << BI->getSuccessor(0)->getName()
                        << " else ->" << BI->getSuccessor(1)->getName() << "\n");
    }
    blockIter++;
  }
}
LLVM_DEBUG(dbgs() << "Find "
                  << LoopNode->getNumNeighbors() << " children\n ");
return LoopNode;
}
*/

void LoopAnalysisWrapperPass::GenerateRefExpr(RefTNode *Node)
{
  TranslateStatus status = SUCCESS;
  string array = Translator->ValueToStringExpr(Node->getBase(), status);
  string subscript_expr = "";
  for (auto Index : Node->getSubscripts()) {
    subscript_expr += ("[" + Translator->ValueToStringExpr(Index, status) + "]");
  }
  if (status != SUCCESS)
    MemoryOpNotSupport += 1;
  Node->setExprString(array, subscript_expr);
}

void LoopAnalysisWrapperPass::GenerateLoopExpr(LoopTNode *Node)
{
  TranslateStatus status = SUCCESS;
  string expr = "(" + Translator->ValueToStringExpr(Node->getInductionPhi(),
                                                    status);
  bool isNegativeTransformed = false;
  ICmpInst *LoopCondInst = BAWP->getBranchAnalyzer().GetLoopBranch(
      Node->getLoop(), isNegativeTransformed);
  if (isNegativeTransformed)
    BranchInversedCount++;
  LoopBound *LB =
      BuildLoopBound(Node->getLoop(), &SEWP->getSE(), LoopCondInst,
                     isNegativeTransformed, Node->getInductionPhi());
  if (LB) {
    Node->setLoopBound(LB);
    expr += ",";
    expr += Translator->ValueToStringExpr(LB->InitValue, status);
    expr += ",\\";
    expr += Translator->PredicateToStringExpr(LB->Predicate, status);
    expr += ",";
    if (!LB->FinalValue) {
      expr += Translator->SCEVToStringExpr(LB->FinalValueSCEV, Node->getLoop(),
                                           status);
    } else {
      expr += Translator->ValueToStringExpr(LB->FinalValue, status);
    }
    expr += ",";
    expr += Translator->ValueToStringExpr(LB->StepInst, status);
  } else {
    LLVM_DEBUG(dbgs() << "No parsable Loop Bound\n");
    status = NOT_TRANSLATEABLE;
  }
  expr += ")";
  if (status != SUCCESS)
    LoopNotSupport += 1;
  if (Node->isParallelLoop() && LB) {
    if (ConstantInt *StepInt = dyn_cast<ConstantInt>(LB->StepValue)) {
      if (StepInt->isNegative())
        ParallelDescLoopCount++;
      else
        ParallelAscLoopCount++;
    }
  }
  Node->setLoopStringExpr(expr);
}

void LoopAnalysisWrapperPass::GenerateBranchExpr(BranchTNode *Node)
{
  TranslateStatus status = SUCCESS;
  string expr = Translator->ValueToStringExpr(Node->getCondition(), status);
  Node->setConditionExpr(expr);

}

void LoopAnalysisWrapperPass::DecorateBranchNode(BranchTNode *BrNode, BasicBlock *TakenBranchHeader,
                        BasicBlock *TakenBranchTerminator, bool isTrue)
{
  SmallVector<Path, 16> Pathes;
  FindAllPathesBetweenTwoBlock(TakenBranchHeader, TakenBranchTerminator, Pathes);
  if (Pathes.size() > 1) {
    LLVM_DEBUG(
        dbgs() << TakenBranchHeader->getName() << " has sub branches\n";
    );
    // we merge the basicblocks inside these path. Those common blocks are
    // the one that has a branch instruction.
    // we traverse those common nodes in order, if the node is normal, we
    // attach all references inside, otherwise, we call BuildBranchNodeFor()
    // with the BI found in that node;
    map<BasicBlock *, unsigned> visited;
    for (auto path:  Pathes) {
      for (auto Block : path) {
        if (visited.find(Block) == visited.end())
          visited[Block] = 1;
        else
          visited[Block] ++;
      }
    }
    for (auto mapEntry : visited) {
      if (mapEntry.second == Pathes.size()) {
        BranchInst *SubBranch = BAWP->getBranchAnalyzer().GetBranchInsideBlock(mapEntry.first);
        if (SubBranch) {
          SPSTNode *SubBranchNode = BuildBranchNodeFor(SubBranch);
          if (isTrue)
            BrNode->addTrueNode(SubBranchNode);
          else
            BrNode->addFalseNode(SubBranchNode);
        } else {
          vector<SPSTNode *> RefNodes;
          AddRefNodeToList(mapEntry.first, RefNodes);
          vector<SPSTNode *>::iterator refIter = RefNodes.begin();
          while (refIter != RefNodes.end()) {
            if (isTrue)
              BrNode->addTrueNode(*refIter);
            else
              BrNode->addFalseNode(*refIter);
            refIter++;
          }
          AccessNodeCount += RefNodes.size();
        }

      }
    }
  } else {
    // Only one path inside this condition, we traverse each block inside
    // this path and construct a list of RefNodes.
    for (auto Block : Pathes.front()) {
      if (Block == TakenBranchTerminator)
        continue;
      vector<SPSTNode *> RefNodes;
      AddRefNodeToList(Block, RefNodes);
      vector<SPSTNode *>::iterator refIter = RefNodes.begin();
      while (refIter != RefNodes.end()) {
        if (isTrue)
          BrNode->addTrueNode(*refIter);
        else
          BrNode->addFalseNode(*refIter);
        refIter++;
      }
      AccessNodeCount += RefNodes.size();
    }
  }
}

SPSTNode *LoopAnalysisWrapperPass::BuildBranchNodeFor(BranchInst *BI)
{
  BasicBlock *IPostDom = ImmediatePostDominator(
      getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree(),
      BI->getParent());
  if (!IPostDom)
    return nullptr;
  assert(IPostDom && "The Immediate post dominator of the branch block cannot"
                     "be null");
  assert(BI->getNumSuccessors() == 2 && "The branch instruction should "
                                        "have two successors");
  BranchTNode *BranchNode = new BranchTNode(BI->getCondition());
  GenerateBranchExpr(BranchNode);
  BasicBlock *TrueBlock = BI->getSuccessor(0);
  BasicBlock *FalseBlock = BI->getSuccessor(1);



  // We use the same steps to process the if and else branch
  DecorateBranchNode(BranchNode, TrueBlock, IPostDom, true);
  DecorateBranchNode(BranchNode, FalseBlock, IPostDom, false);
  return BranchNode;
}



void LoopAnalysisWrapperPass::FindAllInductionVarDependencyInLoop(Loop *L)
{
  stack<pair<IVDepNode *, unsigned>> LoopStack;
  unsigned level = 0;
  IVDepNode *ToplevelNode = new IVDepNode(L,
                                          getInductionVariable(L, SEWP->getSE()),
                                          level);
  LoopStack.push(make_pair(ToplevelNode, level));
  // key: loop , value:its immediate parent
  unordered_map<IVDepNode *, IVDepNode *> ParentMapping;
  while (!LoopStack.empty()) {
    IVDepNode *TopNode = LoopStack.top().first;
    level = LoopStack.top().second;
    LoopStack.pop();
    Loop *TopLoop = TopNode->getLoop();
    for (auto SubL : TopLoop->getSubLoops()) {
      IVDepNode *NextlevelNode = new IVDepNode(SubL,
                                               getInductionVariable(SubL, SEWP->getSE()),
                                               level+1);
      LoopStack.push(make_pair(NextlevelNode, level+1));
      ParentMapping[NextlevelNode] = TopNode;
      // now we check all loops that contains the SubL
      IVDepNode *ParentIter = NextlevelNode;
      do {
        ParentIter = ParentMapping[ParentIter];
        Loop *ParentLoop = ParentIter->getLoop();
        DIRECTION type = hasDependency(ParentLoop, SubL);
        if (type != NO_DEP) {
          LLVM_DEBUG(dbgs() << "Find ";
                     if (type == LOWER_BOUND_DEP)
                         dbgs() << "lower bound ";
                     else if (type == UPPER_BOUND_DEP)
                         dbgs() << "upper bound ";
                     else if (type == DUAL_DEP)
                         dbgs() << "lower and upper bound";
                     dbgs() << "dependences\n";
                     );
          ParentIter->addDependency(NextlevelNode, type);
          NextlevelNode->addParent(ParentIter, type);
          if (InductionVarDepForest.find(ParentIter) == InductionVarDepForest.end())
            InductionVarDepForest.insert(ParentIter);
        }
      } while (ParentIter->getLoop() != L);
    }
  }
}

DIRECTION LoopAnalysisWrapperPass::hasDependency(Loop *Parent, Loop *Child)
{
  assert(Parent && Child && "Both Parent and Child loop should exist");
  assert(Parent != Child && "Child and Parent should not be equal");
  bool NeedInversePredicate = false;
  ICmpInst *ChildLoopBranch = BAWP->getBranchAnalyzer().GetLoopBranch(Child, NeedInversePredicate);
  PHINode *ChildInductionPhi = getInductionVariable(Child, SEWP->getSE());
  PHINode *ParentInductionPhi = getInductionVariable(Parent, SEWP->getSE());
  LoopBound *ChildLB = BuildLoopBound(Child, &SEWP->getSE(), ChildLoopBranch, NeedInversePredicate,
                                      ChildInductionPhi);

  LLVM_DEBUG(
      dbgs() << "Child " << *ChildInductionPhi << "\n";
      dbgs() << " -- Init: " << *(ChildLB->InitValue) << "\n";
      if (ChildLB->FinalValue)
        dbgs() << " -- Final: " << *(ChildLB->FinalValue) << "\n";
      else if (ChildLB->FinalValueSCEV)
        dbgs() << " -- Final: " << *(ChildLB->FinalValueSCEV) << "\n";
  );
  // we here use a trivial approach
  // we first translate the string representation of the loop init and final value
  // then we check if any of the string contains the name of the child loop induction
  // variable
  TranslateStatus  status = SUCCESS;
  string ParentInductionStr = Translator->ValueToStringExpr(ParentInductionPhi, status);
  string ChildInitValueStr = Translator->ValueToStringExpr(ChildLB->InitValue, status);
  string ChildFinalValueStr = Translator->ValueToStringExpr(ChildLB->FinalValue, status);
  LLVM_DEBUG(
      dbgs() << "Check " << ParentInductionStr << " in " << ChildInitValueStr << " or "
      << ChildFinalValueStr << "\n";
      );
  if (status != SUCCESS)
    return NO_DEP;
//  if (ChildInitValueStr.find(ParentInductionStr) != string::npos) {
//    return LOWER_BOUND_DEP;
//  } else if (ChildFinalValueStr.find(ParentInductionStr) != string::npos) {
//    return UPPER_BOUND_DEP;
//  }
  DIRECTION deptype = NO_DEP;
  if (FindValueInInstruction(ChildLB->InitValue, ParentInductionPhi)) {
    deptype = LOWER_BOUND_DEP;
    if (ChildLB->FinalValue && FindValueInInstruction(ChildLB->FinalValue, ParentInductionPhi)) {
      deptype = DUAL_DEP;
    } else if (FindValueInSCEV(ChildLB->FinalValueSCEV, ParentInductionPhi, SEWP->getSE())) {
      deptype = DUAL_DEP;
    }
  } else if (ChildLB->FinalValue && FindValueInInstruction(ChildLB->FinalValue, ParentInductionPhi)) {
    deptype = UPPER_BOUND_DEP;
  } else if (FindValueInSCEV(ChildLB->FinalValueSCEV, ParentInductionPhi, SEWP->getSE())) {
    deptype = UPPER_BOUND_DEP;
  }
  return deptype;
}


void LoopAnalysisWrapperPass::DumpTree()
{
  queue<pair<SPSTNode *, unsigned>> q;
  unsigned level = 0;
  q.push(make_pair(TreeRoot, 0));
  while (!q.empty()) {
    SPSTNode *top = q.front().first;
    level = q.front().second;
    q.pop();
    string nodeclass = "SPSTNode";
    if (RefTNode *Node = dynamic_cast<RefTNode*>(top)) {
      Node->dump();
    } else if (ThreadTNode *Node = dynamic_cast<ThreadTNode*>(top)) {
    } else if (BranchTNode *Node = dynamic_cast<BranchTNode*>(top)) {
      Node->dump();
    } else if (LoopTNode *Node = dynamic_cast<LoopTNode*>(top)) {
      Node->dump();
      assert(Node->getLoopLevel() == (level - 1) && "Loop level does not match");
    } else if (DummyTNode *Node = dynamic_cast<DummyTNode*>(top)) {
    }
    auto neighbor_iter = top->neighbors.begin();
    while (neighbor_iter != top->neighbors.end()) {
      SPSTNode *tmp = *neighbor_iter;
      if (tmp)
        q.push(make_pair(tmp, level+1));
      neighbor_iter++;
    }
  }
}


void LoopAnalysisWrapperPass::ViewIndudctionVarDependency()
{
  string tab = "\t";
  string nodeclass = "IVDepNode", shape = "record", content = "", fillcolor="none";
  string code = "digraph \"InductionVariable DependencyTree \" {\n";
  code += (tab + "label=\"InductionVariable DependencyTree;\"\n");
  auto deprootIter = InductionVarDepForest.begin();
  while (deprootIter != InductionVarDepForest.end()) {
    IVDepNode *parent = *(deprootIter);
    for (auto p : parent->dependences) {
      IVDepNode *child = p.first;
      string child_addr = "", parent_addr = "";
      if (p.second == LOWER_BOUND_DEP)
        fillcolor = "lightcoral";
      else if (p.second == UPPER_BOUND_DEP)
        fillcolor = "lightblue1";
      else if (p.second == DUAL_DEP)
        fillcolor = "khaki";
      stringstream ss;
      ss << child;
      child_addr = ss.str();
      ss.str("");
      ss << parent;
      parent_addr = ss.str();
      TranslateStatus status;
      if (child->getLoop())
        content = Translator->ValueToStringExpr(child->getInduction(), status)
                  + ", "
                  + to_string(child->getLevel());
      code += (tab + "Node" + child_addr + " " + "[fillcolor=\"" + fillcolor + "\","
               + "style=\"rounded,filled\",shape=" + shape + ","
               + "label=\"{" + nodeclass + ":\\l  " + content + "\\l}\"];\n");
      if (parent->getLoop())
        content = Translator->ValueToStringExpr(parent->getInduction(), status)
                  + ", "
                  + to_string(parent->getLevel());
      code += (tab + "Node" + parent_addr + " " + "[fillcolor=\"" + fillcolor + "\","
               + "style=\"rounded,filled\",shape=" + shape + ","
               + "label=\"{" + nodeclass + ":\\l  " + content + "\\l}\"];\n");
      code += tab + "Node" + child_addr + " -> ";
      code += "Node" + parent_addr + ";\n";
    }
    deprootIter++;
  }
  code += "}\n";
  LLVM_DEBUG(dbgs() << code << "\n");
}


void LoopAnalysisWrapperPass::ViewLoopTree(SPSTNode *Root, string title)
{
  string tab = "\t";
  string code = "digraph \"AbstractionTree for '" + title + "' function\" {\n";
  code += (tab + "label=\"CFG for '" + title +
           "' function\";\n");
  queue<pair<SPSTNode *, string>> q;
  q.push(make_pair(Root, tab));
  while (!q.empty()) {
    SPSTNode *top = q.front().first;
    q.pop();
    stringstream ss;
    ss << top;
    string nodeclass = "SPSTNode", content = "";
    string fillcolor = "invis", shape = "record";
    if (RefTNode *Node = dynamic_cast<RefTNode*>(top)) {
      nodeclass = "RefTNode";
      if (!Node->getBase()) {
        fillcolor = "red";
      } else {
        content = Node->getRefExprString();
      }
      shape = "ellipse";
    } else if (ThreadTNode *Node = dynamic_cast<ThreadTNode*>(top)) {
      nodeclass = "ThreadTNode";
    } else if (BranchTNode *Node = dynamic_cast<BranchTNode*>(top)) {
      nodeclass = "BranchTNode";
      content = Node->getConditionExpr();
      fillcolor = "aquamarine";
    } else if (LoopTNode *Node = dynamic_cast<LoopTNode*>(top)) {
      nodeclass = "LoopTNode";
      content = "["+to_string(Node->getLoopLevel())+"]: "+Node->getLoopStringExpr();
      fillcolor = "lemonchiffon";
      if (Node->isParallelLoop())
        fillcolor = "plum";
    } else if (DummyTNode *Node = dynamic_cast<DummyTNode*>(top)) {
      nodeclass = Node->getName();
    }
    code += (tab + "Node" + ss.str() + " " + "[fillcolor=\"" + fillcolor + "\","
             + "style=\"rounded,filled\",shape=" + shape + ","
             + "label=\"{" + nodeclass + ":\\l  " + content + "\\l}\"];\n");
    string parent = ss.str();
    auto neighbor_iter = top->neighbors.begin();
    while (neighbor_iter != top->neighbors.end()) {
      SPSTNode *tmp = *neighbor_iter;
      code += tab + "Node" + parent + " -> ";
      ss.str("");
      ss << tmp;
      code += "Node" + ss.str() + ";\n";
      if (tmp)
        q.push(make_pair(tmp, tab+"\t"));
      neighbor_iter++;
    }
  }
  code += "}\n";
  LLVM_DEBUG(dbgs() << code << "\n");
}

bool LoopAnalysisWrapperPass::IsPlussParallelLoop(Loop *L)
{
  MDNode *LoopMD = L->getLoopID();
  if (!LoopMD) {
    LLVM_DEBUG(dbgs() << "Loop " << L->getHeader()->getName() << " does not have Loop MDNode\n");
    // special case, when the loop is simplified by the LoopSimplify pass, a loopexist
    // block will be inserted and the looplatch will return this loopexit block.
    // in this case, the parallel attribute is at the terminator of the immediate
    // pred of the loopexit block
    if (L->getSubLoops().empty())
      return false;
    SmallVector<BasicBlock *, 8> LoopLatches;
    L->getLoopLatches(LoopLatches);
    LLVM_DEBUG(
        dbgs() << "Check parallel for loop " << L->getHeader()->getName() << "\n";
        for (auto block : LoopLatches) {
          dbgs() << "Latches " << block->getName() << "\n";
        }
    );
    for (auto latch : LoopLatches) {
      BasicBlock *SinglePred = latch->getSinglePredecessor();
      if (SinglePred) {
        Instruction *TI = latch->getSinglePredecessor()->getTerminator();
        LLVM_DEBUG(if (TI) dbgs() << "Check predecessor "
                                  << latch->getSinglePredecessor()->getName()
                                  << " " << *TI << "\n");
        MDNode *MD = TI->getMetadata(LLVMContext::MD_loop);
        if (!MD)
          return false;

        if (!LoopMD)
          LoopMD = MD;
        else if (MD != LoopMD)
          return false;
      }
    }
    if (!LoopMD || LoopMD->getNumOperands() == 0 ||
        LoopMD->getOperand(0) != LoopMD)
      return false;
#if 0
    unordered_set<BasicBlock *> SubLoopBlocks;
    getBasicBlockInAllSubLoops(L, SubLoopBlocks, true);
    for (auto block : L->getBlocksVector()) {
      if (SubLoopBlocks.find(block) == SubLoopBlocks.end()) {
        LLVM_DEBUG(dbgs() << block->getName() << " is in Loop " << L->getHeader()->getName() << " but not in its subloops\n");
        Instruction *TI = block->getTerminator();
        MDNode *MD = TI->getMetadata(LLVMContext::MD_loop);
        if (!MD)
          continue;
        if (!LoopMD)
          LoopMD = MD;
      }
    }
    if (!LoopMD || LoopMD->getNumOperands() == 0 ||
        LoopMD->getOperand(0) != LoopMD) {
      LLVM_DEBUG(dbgs() << "Loop " << L->getHeader()->getName() << " does not have Loop MDNode\n");
      return false;
    }
#endif
  }
  return GetMetadataWithName(LoopMD, "llvm.loop.pluss.parallel");
}


/// Call for every function
bool LoopAnalysisWrapperPass::runOnFunction(Function &F)
{
  if (!FunctionName.empty() && F.getName().str().find(FunctionName) == string::npos) {
    // the function name is given, only the given function will be analyzed
    LLVM_DEBUG(dbgs() << F.getName() << " will be skipped\n");
    return false;
  }
  LIWP = &getAnalysis<LoopInfoWrapperPass>();
  SEWP = &getAnalysis<ScalarEvolutionWrapperPass>();
  BAWP = &getAnalysis<BranchAnalysis::BranchAnalysisWrapperPass>();
  Translator = new StringTranslator(
      &SEWP->getSE(),
      getAnalysis<InductionVarAnalysis::InductionVarAnalysis>()
          .InductionVarNameTable);
#if defined(__APPLE__) && defined(__MACH__)
  LLVM_DEBUG(dbgs() << __FILE_NAME__ << " on " << F.getName() << "\n");
#elif defined(__linux__)
  LLVM_DEBUG(dbgs() << __FILE__ << " on " << F.getName() << "\n");
#endif
  LoopInfo *LI = &(LIWP->getLoopInfo());
  LoopCnt = 0;
  DummyTNode *DummyRoot = new DummyTNode("Root");
  // We first add all LoopNode to the tree
  unordered_set<BasicBlock *> VisitedBasicBlock;
  auto blockIter = F.begin();
  auto toploopIter = LI->getTopLevelLoops().rbegin();
  while (blockIter != F.end()) {
    if (!LI->getTopLevelLoops().empty() &&
        toploopIter != LI->getTopLevelLoops().rend() &&
        (*toploopIter)->contains(&*blockIter)) {
      // if we found a block that belongs to a subloop, we build the tree node
      // for the subloop, and jump all other blocks belonging to this subloop.
      // Then we move to the next subloop.
      // subloops in L->getSubLoop() vector follows the topology order,  so the traversal always follows the program flow.
      Loop *TopL = *toploopIter;
      BuildTreeForLoopImpl(TopL, VisitedBasicBlock, 0);
      AppendRefNodesOnTreeImpl(LoopToNodeMapping[TopL]);
      AppendBranchNodesOnTreeImpl(LoopToNodeMapping[TopL]);
      DummyRoot->addNeighbors(LoopToNodeMapping[TopL]);
      while (TopL->contains(&*blockIter)) {
        blockIter++;
      }
      toploopIter++; // we move to the next subloop
    } else if (VisitedBasicBlock.find(&*blockIter) !=
               VisitedBasicBlock.end()) {
      LLVM_DEBUG(dbgs() << "Skip block " << (*blockIter).getName() << " \n");
      blockIter++;
    } else {
      VisitedBasicBlock.insert(&*blockIter);
      vector<SPSTNode *> RefNodes;
      vector<SPSTNode *>::iterator refIter;
      AddRefNodeToList(&*blockIter, RefNodes);
      refIter = RefNodes.begin();
      while (refIter != RefNodes.end()) {
        DummyRoot->addNeighbors(*refIter);
        refIter++;
      }
      blockIter++;
      AccessNodeCount += RefNodes.size();
    }
  }
  TreeRoot = DummyRoot;

  for (auto TopLoop : LI->getTopLevelLoops()) {
    FindAllInductionVarDependencyInLoop(TopLoop);
  }
  //  F.viewCFG();
  if (ViewAbstractionTree)
    ViewLoopTree(TreeRoot, F.getName().str());

  if (ViewInductionVariableDependency)
    ViewIndudctionVarDependency();
  return false;
}

void LoopAnalysisWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const
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
  AU.addRequired<BranchAnalysis::BranchAnalysisWrapperPass>();
}


} // end of LoopAnalysisWrapperPass namespace