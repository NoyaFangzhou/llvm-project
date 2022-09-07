//
//  LoopAnalysisUtils.cpp
//  LLVMSPSAnalysis
//
//  Created by noya-fangzhou on 10/14/21.
//

#include "LoopAnalysisUtils.h"
#define DEBUG_TYPE "loop-analysis-utils"

LoopTNode::LoopTNode(Loop *L, unsigned LL, unsigned id,
                     ScalarEvolution &SE) : SPSTNode()
{
  this->L = L;
  this->IsParallelLoop = false;
  this->LoopLevel = LL;
  this->ID = id;
  this->name = "LoopTNode";
  if (L->getInductionVariable(SE))
    this->InductionVariable = L->getInductionVariable(SE);
  else if (L->getCanonicalInductionVariable())
    this->InductionVariable = L->getCanonicalInductionVariable();
  else if (getInductionVariable(L, SE))
    this->InductionVariable = getInductionVariable(L, SE);
  else if (getInductionVariableV2(L, SE))
    this->InductionVariable = getInductionVariableV2(L, SE);
}

LoopTNode::LoopTNode(Loop *L, unsigned LL, unsigned id, bool IsParallel,
                     ScalarEvolution &SE) : SPSTNode()
{
  this->L = L;
  this->IsParallelLoop = IsParallel;
  this->ID = id;
  this->LoopLevel = LL;
  this->SE = &SE;
  this->name = "LoopTNode";
  if (L->getInductionVariable(SE))
    this->InductionVariable = L->getInductionVariable(SE);
  else if (L->getCanonicalInductionVariable())
    this->InductionVariable = L->getCanonicalInductionVariable();
  else if (getInductionVariable(L, SE))
    this->InductionVariable = getInductionVariable(L, SE);
  else if (getInductionVariableV2(L, SE))
    this->InductionVariable = getInductionVariableV2(L, SE);
}

void LoopTNode::dump()
{
  dbgs() << this->LoopStringExpr << "\n";
}

RefTNode::RefTNode(Instruction *Ref, GetElementPtrInst *GEP,
                   ScalarEvolution &SE)  : SPSTNode()
{
  this->MemOp = Ref;
  this->ArrayAccess = GEP;
  this->name = "RefTNode";
  if (GEP) {
    unsigned dimension = GEP->getNumIndices();
    for (User *U : GEP->getOperand(0)->users()) {
      if (GetElementPtrInst *GEPUser = dyn_cast<GetElementPtrInst>(U))
        dimension = max(dimension, GEPUser->getNumIndices());
    }
    LLVM_DEBUG(dbgs() << *(GEP->getOperand(0)) << " is a " << dimension << " dimension array\n");
    // a[i][j]
    if (GEP->getNumIndices() == dimension) {
      for (unsigned subIdx = 1; subIdx < GEP->getNumOperands(); subIdx++) {
        Subscripts.push_back(GEP->getOperand(subIdx));
      }
      this->Base = GEP->getOperand(0);
    } else {
      // a[i] or **a or a[i][0]
      SmallVector<Value *, 8> TempSubscripts;
      TempSubscripts.push_back(GEP->getOperand(1));
      this->Base = FindBaseAndSubscript(GEP->getOperand(0), TempSubscripts);
      if (!this->Base || TempSubscripts.empty()) {
        LLVM_DEBUG(dbgs() << "This memory access is not supported");
      } else {
        auto subrIter = TempSubscripts.rbegin();
        while (subrIter != TempSubscripts.rend()) {
          Subscripts.push_back(*subrIter);
          subrIter++;
        }
      }
      for (unsigned diff = 0; diff < (dimension - GEP->getNumIndices()); diff++) {
        // add 0 indices
        ConstantInt *ZeroIndex = ConstantInt::get(Ref->getContext(), APInt(/*nbits*/32, 0, /*bool*/false));
        Subscripts.push_back(ZeroIndex);
      }
    }
  } else {
    unsigned dimension = 0;
    unsigned array_pointer_idx = 0;
    if (isa<StoreInst>(Ref)) {
      array_pointer_idx = 1;
    }
    for (User *U : Ref->getOperand(array_pointer_idx)->users()) {
      if (GetElementPtrInst *GEPUser = dyn_cast<GetElementPtrInst>(U))
        dimension = max(dimension, GEPUser->getNumIndices());
    }
    for (unsigned diff = 0; diff < dimension; diff++) {
      ConstantInt *ZeroIndex = ConstantInt::get(
          Ref->getContext(), APInt(/*nbits*/ 32, 0, /*bool*/ false));
      Subscripts.push_back(ZeroIndex);
    }
    this->Base = Ref->getOperand(array_pointer_idx);
  }

}

void RefTNode::dump()
{
  dbgs() << this->arrayName << this->Expression << "\n";
}

Value *RefTNode::FindBaseAndSubscript(Value *V,
                                      SmallVectorImpl<Value *> &Subscripts)
{
  // we assume V can be
  // 1) GetElementPtr // contains the subscript info
  // 2) LoadInst // could load one row from a multidimension array
  // 3) Argument
  // 4) AllocaInst
  if (!V || isa<Argument>(V))
    return V;
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return nullptr;
  switch (I->getOpcode()) {
  case Instruction::Load: {
    LoadInst *Load = dyn_cast<LoadInst>(I);
    return FindBaseAndSubscript(Load->getOperand(0), Subscripts);
  }
  case Instruction::Alloca: {
    return V;
  }
  case Instruction::GetElementPtr: {
    GetElementPtrInst *GepInst = dyn_cast<GetElementPtrInst>(I);
    Subscripts.push_back(GepInst->getOperand(1));
    return FindBaseAndSubscript(GepInst->getOperand(0), Subscripts);
  }
  default:
    break;
  }
  return nullptr;

}

void getBasicBlockInAllSubLoops(Loop *L,
                                unordered_set<BasicBlock *> &SubLoopBasicBlocks,
                                bool ExcludeL)
{
  stack<Loop *> LoopNests;
  LoopNests.push(L);
  bool isHead = ExcludeL;
  while (!LoopNests.empty()) {
    Loop *Head = LoopNests.top();
    LoopNests.pop();
    if (!isHead) {
      for (auto BB : Head->getBlocks()) {
        SubLoopBasicBlocks.insert(BB);
      }
    } else {
      isHead = false;
    }
    for (auto SubL : Head->getSubLoops()) {
      LoopNests.push(SubL);
    }
  }
}

void getBasicBlockInAllSubLoopsInOrder(Loop *L,
                                vector<BasicBlock *> &SubLoopBasicBlocks,
                                bool ExcludeL) {
  stack<Loop *> LoopNests;
  LoopNests.push(L);
  bool isHead = ExcludeL;
  while (!LoopNests.empty()) {
    Loop *Head = LoopNests.top();
    LoopNests.pop();
    if (!isHead) {
      for (auto BB : Head->getBlocks()) {
        SubLoopBasicBlocks.push_back(BB);
      }
    } else {
      isHead = false;
    }
    for (auto SubL : Head->getSubLoops()) {
      LoopNests.push(SubL);
    }
  }
}


LoopBound *BuildLoopBound(Loop *L, ScalarEvolution *SE, ICmpInst *LoopBranch,
                          bool NeedInversePredicate, PHINode *IndVar)
{
  if (LoopBranch)
    LLVM_DEBUG(dbgs() << "Branch " << *LoopBranch << " \n");
  if (!IndVar || !LoopBranch)
    return nullptr;
  InductionDescriptor IndDesc;
  if (!InductionDescriptor::isInductionPHI(IndVar, L, SE, IndDesc))
    return nullptr;
//  const SCEVAddRecExpr *InductionPhiSCEV =
//      dyn_cast<SCEVAddRecExpr>(SE->getSCEV(IndVar));

//  InductionPhiSCEV->getStart();

  Value *InitialIVValue = IndDesc.getStartValue();
  Instruction *StepInst = IndDesc.getInductionBinOp();
  if (!InitialIVValue || !StepInst)
    return nullptr;

  const SCEV *Step = IndDesc.getStep();
  Value *StepInstOp1 = StepInst->getOperand(1);
  Value *StepInstOp0 = StepInst->getOperand(0);
  Value *StepValue = nullptr;
  if (SE->getSCEV(StepInstOp1) == Step)
    StepValue = StepInstOp1;
  else if (SE->getSCEV(StepInstOp0) == Step)
    StepValue = StepInstOp0;
  if (!StepValue)
    return nullptr;
  Value *FinalValue = nullptr;
  Value *Op0 = LoopBranch->getOperand(0);
  Value *Op1 = LoopBranch->getOperand(1);

  IVPos pos = LHS;

  if (Op0 == IndVar || Op0 == StepInst) {
    FinalValue = Op1;
  }
  if (Op1 == IndVar || Op1 == StepInst) {
    FinalValue = Op0;
    pos = RHS;
  }

  CmpInst::Predicate LoopConditionPredicate = LoopBranch->getPredicate();
  if (NeedInversePredicate)
    LoopConditionPredicate = LoopBranch->getInversePredicate();

  // if the FinalValue is given from the #pragma pluss bound
  if (GetLoopMetadataConstValue(L->getLoopID()) != 0) {
    LLVM_DEBUG(dbgs() << "Has metadata constant " << GetLoopMetadataConstValue(L->getLoopID()) << "\n");
    unsigned FinalValue = GetLoopMetadataConstValue(L->getLoopID());
    return new LoopBound(InitialIVValue,
                         SE->getConstant(Type::getInt32Ty(L->getHeader()->getContext()),
                                         FinalValue),
                         StepValue, StepInst, LoopConditionPredicate, nullptr, pos);
//  } else if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
//    const SCEV *FinalValueSCEV = SE->getBackedgeTakenCount(L);
//    return new LoopBound(InitialIVValue, FinalValueSCEV, StepValue, StepInst,
//                         LoopBranch->getPredicate());
  }

  if (FinalValue) {
    const SCEV *FinalValueSCEV = SE->getSCEV(FinalValue);
    return new LoopBound(InitialIVValue, FinalValueSCEV, StepValue, StepInst,
                         LoopConditionPredicate, FinalValue, pos);
  } else {
    LLVM_DEBUG(dbgs() << "The upper bound of the loop is not computable\n");
  }
  return nullptr;

}

/// Given an llvm.loop loop id metadata node, returns the pluss hint metadata
/// node with the given name (for example, "llvm.loop.pluss.parallel"). If no
/// such metadata node exists, then nullptr is returned.
MDNode *GetMetadataWithName(MDNode *LoopMD, StringRef Name) {
  // First operand should refer to the loop id itself.
  assert(LoopMD->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopMD->getOperand(0) == LoopMD && "invalid loop id");

  for (unsigned i = 1, e = LoopMD->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopMD->getOperand(i));
    if (!MD) {
      LLVM_DEBUG(dbgs() << "LoopMD[" << i << "] is null" << "\n");
      continue;
    }

    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S) {
      LLVM_DEBUG(dbgs() << "LoopMD[" << i << "] is not a MDString" << "\n");
      continue;
    }
    LLVM_DEBUG(dbgs() << "MDString " << S->getString() << "\n");
    if (Name.equals(S->getString()))
      return MD;
  }
  return nullptr;
}


// If loop has an pluss pragma with a integer value, i.e. the bound pragma
// return the (necessarily positive) value from the pragma.
// Otherwise return 0.
unsigned GetLoopMetadataConstValue(MDNode *LoopMD, StringRef name) {
  if (!LoopMD)
    return 0;
  MDNode *MD = GetMetadataWithName(LoopMD, name);
  if (MD) {
    assert(MD->getNumOperands() >= 2 &&
           "pluss loop hint metadata should have two operands.");
    unsigned Count =
        mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();
    assert(Count >= 1 && "pragma value must be positive.");
    return Count;
  }
  return 0;
}

Loop *FindSubloopContainsBlockInLoop(Loop *ParentLoop, BasicBlock *Target)
{
  for (auto SubL : ParentLoop->getSubLoops()) {
    if (SubL->contains(Target))
      return SubL;
  }
  return nullptr;
}

void FindSubLoopsContainBlock(Loop *ParentLoop, BasicBlock *Target,
                              vector<Loop *> &SubLoops)
{
  for (auto SubL : ParentLoop->getSubLoops()) {
    if (SubL->contains(Target)) {
      SubLoops.push_back(SubL);
    }
    FindSubLoopsContainBlock(SubL, Target, SubLoops);
  }
}

/*
PHINode *getInductionVariableV2(Loop *L, ScalarEvolution &SE) {
  if (!L->isLoopSimplifyForm())
    return nullptr;

  BasicBlock *Header = L->getHeader();
  assert(Header && "Expected a valid loop header");

  ICmpInst *CmpInst = nullptr;
  if (BasicBlock *Latch = L->getLoopLatch())
    if (BranchInst *BI = dyn_cast_or_null<BranchInst>(Latch->getTerminator()))
      if (BI->isConditional())
        CmpInst = dyn_cast<ICmpInst>(BI->getCondition());
  if (!CmpInst) {
    if (BranchInst *BI = dyn_cast_or_null<BranchInst>(Header->getTerminator()))
      if (BI->isConditional())
        CmpInst = dyn_cast<ICmpInst>(BI->getCondition());
  }

  if (!CmpInst)
    return nullptr;

  Instruction *CmpOp0 = dyn_cast<Instruction>(CmpInst->getOperand(0));
  Instruction *CmpOp1 = dyn_cast<Instruction>(CmpInst->getOperand(1));

  for (PHINode &IndVar : Header->phis()) {
    InductionDescriptor IndDesc;
    if (!InductionDescriptor::isInductionPHI(&IndVar, L, &SE, IndDesc))
      continue;

    Instruction *StepInst = IndDesc.getInductionBinOp();

    // case 1:
    // IndVar = phi[{InitialValue, preheader}, {StepInst, latch}]
    // StepInst = IndVar + step
    // cmp = StepInst < FinalValue
    if (StepInst == CmpOp0 || StepInst == CmpOp1)
      return &IndVar;

    // case 2:
    // IndVar = phi[{InitialValue, preheader}, {StepInst, latch}]
    // StepInst = IndVar + step
    // cmp = IndVar < FinalValue
    if (&IndVar == CmpOp0 || &IndVar == CmpOp1)
      return &IndVar;
  }

  return nullptr;
}

PHINode *getInductionVariable(Loop *L, ScalarEvolution &SE)
{
  PHINode *InnerIndexVar = L->getCanonicalInductionVariable();
  if (InnerIndexVar)
    return InnerIndexVar;
  if (L->getLoopLatch() == nullptr || L->getLoopPredecessor() == nullptr)
    return nullptr;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    PHINode *PhiVar = cast<PHINode>(I);
    Type *PhiTy = PhiVar->getType();
    if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
        !PhiTy->isPointerTy())
      return nullptr;
    const SCEVAddRecExpr *AddRec =
        dyn_cast<SCEVAddRecExpr>(SE.getSCEV(PhiVar));
    if (!AddRec || !AddRec->isAffine())
      continue;
    const SCEV *Step = AddRec->getStepRecurrence(SE);
    if (!isa<SCEVConstant>(Step))
      continue;
    // Found the induction variable.
    // FIXME: Handle loops with more than one induction variable. Note that,
    // currently, legality makes sure we have only one induction variable.
    return PhiVar;
  }
  return nullptr;

}
 */

TreeStructureQueryEngine::TreeStructureQueryEngine(SPSTNode *Root)
{
  this->Root = Root;
  this->TreeDegree = 0;
  BuildParentMapping(this->Root);
  LLVM_DEBUG(dbgs() << "The tree has " << ParentNodeMapping.size() << " pairs\n" );
  stack<SPSTNode *> stack;
  stack.push(Root);
  while (!stack.empty()) {
    SPSTNode *Top = stack.top();
    stack.pop();
    if (dynamic_cast<LoopTNode *>(Top)) {
      SmallVector<SPSTNode *, 8> FirstRefs, LastRefs;
      FindFirstAccessNodesOfParent(Top, FirstRefs);
      FindLastAccessNodesOfParent(Top, LastRefs);
      this->FirstAccesses[Top] = FirstRefs;
      this->LastAccesses[Top] = LastRefs;
      auto neighbor_iter = Top->neighbors.begin();
      for (; neighbor_iter != Top->neighbors.end(); neighbor_iter++) {
        if (LoopTNode *L = dynamic_cast<LoopTNode *>(*neighbor_iter)) {
          this->TreeDegree = max(TreeDegree, L->getLoopLevel());
          stack.push(L);
        } else if (BranchTNode *Branch =
                       dynamic_cast<BranchTNode *>(*neighbor_iter)) {
          stack.push(Branch);
        }
      }
    } else if (dynamic_cast<BranchTNode *>(Top)) {
      auto branch_neighbor_iter = Top->neighbors[0]->neighbors.begin();
      for (; branch_neighbor_iter != Top->neighbors[0]->neighbors.end(); branch_neighbor_iter++) {
        if (LoopTNode *L = dynamic_cast<LoopTNode *>(*branch_neighbor_iter)) {
          this->TreeDegree = max(TreeDegree, L->getLoopLevel());
          stack.push(L);
        } else if (BranchTNode *Branch =
            dynamic_cast<BranchTNode *>(*branch_neighbor_iter)) {
          stack.push(Branch);
        }
      }
      if (!Top->neighbors[1]->neighbors.empty()) {
        branch_neighbor_iter = Top->neighbors[1]->neighbors.begin();
        for (; branch_neighbor_iter != Top->neighbors[1]->neighbors.end();
             branch_neighbor_iter++) {
          if (LoopTNode *L = dynamic_cast<LoopTNode *>(*branch_neighbor_iter)) {
            this->TreeDegree = max(TreeDegree, L->getLoopLevel());
            stack.push(L);
          } else if (BranchTNode *Branch =
                         dynamic_cast<BranchTNode *>(*branch_neighbor_iter)) {
            stack.push(Branch);
          }
        }
      }
    } else if (dynamic_cast<DummyTNode *>(Root)) {
      auto neighbor_iter = Top->neighbors.begin();
      for (; neighbor_iter != Top->neighbors.end(); neighbor_iter++) {
        if (LoopTNode *L = dynamic_cast<LoopTNode *>(*neighbor_iter)) {
          this->TreeDegree = max(TreeDegree, L->getLoopLevel());
          stack.push(L);
        } else if (BranchTNode *Branch =
            dynamic_cast<BranchTNode *>(*neighbor_iter)) {
          stack.push(Branch);
        }
      }
    }
  }
}

bool TreeStructureQueryEngine::isFirstAccess(SPSTNode *Node)
{
  if (RefTNode *Ref = dynamic_cast<RefTNode *>(Node)) {
    for (auto loop : FirstAccesses) {
      for (auto first : loop.second) {
        if (first == Ref)
          return true;
      }
    }
  }
  return false;
}
bool TreeStructureQueryEngine::isLastAccessInLoop(SPSTNode *Node)
{
  if (RefTNode *Ref = dynamic_cast<RefTNode *>(Node)) {
    for (auto loop : LastAccesses) {
      for (auto last : loop.second) {
        if (last == Ref)
          return true;
      }
    }
  }
  return false;
}
bool TreeStructureQueryEngine::areTwoAccessInSameLevel(SPSTNode *A, SPSTNode *B)
{
  LoopTNode *parentA = nullptr, *parentB = nullptr;
  if (ParentNodeMapping.find(A) != this->ParentNodeMapping.end()) {
    parentA = ParentNodeMapping[A];
  }
  if (ParentNodeMapping.find(B) != ParentNodeMapping.end()) {
    parentB = ParentNodeMapping[B];
  }
  if ((!parentA && parentB) || (parentA && !parentB))
    return false;
  if (!parentA && !parentB)
    return true;
  return (parentA->getLoopLevel() == parentB->getLoopLevel());
}
bool TreeStructureQueryEngine::areTwoAccessInSameLoop(SPSTNode *A, SPSTNode *B)
{
  LoopTNode *parentA = nullptr, *parentB = nullptr;
  if (ParentNodeMapping.find(A) != ParentNodeMapping.end()) {
    parentA = ParentNodeMapping[A];
  } else {
    return false;
  }
  if (ParentNodeMapping.find(B) != ParentNodeMapping.end()) {
    parentB = ParentNodeMapping[B];
  } else {
    return false;
  }
  return parentA == parentB;
}
SmallVector<SPSTNode *, 8> TreeStructureQueryEngine::GetFirstAccessInLoop(SPSTNode *Node)
{
  return FirstAccesses[Node];
}
SmallVector<SPSTNode *, 8> TreeStructureQueryEngine::GetLastAccessInLoop(SPSTNode *Node)
{
  return LastAccesses[Node];
}

void TreeStructureQueryEngine::GetImmediateCommonLoopDominator(SPSTNode *A,
                                                                     SPSTNode *B,
                                           bool isLastAccess,
                                           SmallVectorImpl<SPSTNode *> &Dominators)
{
  AccPath pathToA, pathToB;
  this->GetPath(A, NULL, pathToA);
  this->GetPath(B, NULL, pathToB);
  AccPath ::iterator pathAIter = pathToA.begin();
  AccPath::iterator pathBIter = pathToB.begin();
  int degreeDiff = pathToB.size() - pathToA.size();
  while (degreeDiff != 0) {
    if (degreeDiff > 0) { // B is deeper than A
      pathBIter++;
      degreeDiff--;
    } else if (degreeDiff < 0) { // A is deeper than B
      pathAIter++;
      degreeDiff++;
    }
  }
  // now pathAIter and pathBIter at the same loop level
  while (pathAIter != pathToA.end() && pathBIter != pathToB.end()) {
    if ((*pathAIter) == (*pathBIter)) { // these two loops are the same
      Dominators.push_back(*pathAIter);
      break;
    }
    pathAIter++;
    pathBIter++;
  }
}

LoopTNode *TreeStructureQueryEngine::GetImmdiateLoopDominator(SPSTNode *Node)
{
  if (ParentNodeMapping.find(Node) != ParentNodeMapping.end())
    return ParentNodeMapping[Node];
  return nullptr;
}

LoopTNode *TreeStructureQueryEngine::GetParallelLoopDominator(SPSTNode *Node)
{
  if (!Node)
    return nullptr;
  SPSTNode *Tmp = Node;
  while (Tmp && ParentNodeMapping.find(Tmp) != ParentNodeMapping.end()) {
    errs() << "/* " << ParentNodeMapping[Tmp]->getLoopStringExpr() << ", " << ParentNodeMapping[Tmp]->isParallelLoop() << "*/\n";
    if (ParentNodeMapping[Tmp]->isParallelLoop())
      return ParentNodeMapping[Tmp];
    Tmp = ParentNodeMapping[Tmp];
  }
  return nullptr;
}

bool TreeStructureQueryEngine::isLastLevelAccess(SPSTNode *Node)
{
  if (ParentNodeMapping.find(Node) != ParentNodeMapping.end())
    return ParentNodeMapping[Node]->getLoopLevel() == TreeDegree;
  return TreeDegree == 0;
}


void TreeStructureQueryEngine::FindFirstAccessNodesOfParent(SPSTNode *Node,
                                                            SmallVector<SPSTNode *, 8> &FirstRefs)
{
  if (Node->neighbors.empty() && !dynamic_cast<RefTNode *>(Node)) {
    LLVM_DEBUG(dbgs() << "Parent node does not have any child node\n");
    return;
  }
  // if the node is a loop node, we analyze its first neighbor
  // it can be:
  // 1) the first access node of its first subloop
  // 2) the first access node of all branches
  // 3) the first access node
  auto first_neighbor = Node->neighbors.begin();
  if (LoopTNode *L = dynamic_cast<LoopTNode *>(Node)) {
    if (LoopTNode *FirstSubL = dynamic_cast<LoopTNode *>(*first_neighbor)) {
      FindFirstAccessNodesOfParent(FirstSubL, FirstRefs);
    } else if (RefTNode *FirstRef = dynamic_cast<RefTNode *>(*first_neighbor)) {
      FirstRefs.push_back(FirstRef);
    } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(*first_neighbor)) {
      FirstRefs.push_back(Branch);
//      FindFirstAccessNodesOfParent(Branch->neighbors[0], FirstRefs);
//      FindFirstAccessNodesOfParent(Branch->neighbors[1], FirstRefs);
    }
  } else if (RefTNode *Ref = dynamic_cast<RefTNode *>(Node)) {
    // this can be reached if and only if
    // 1) check one branch condition
    // 2) root
    FirstRefs.push_back(Ref);
  } else {
    // the dummy node can be one of the branch condition
    // its first access is the first access of its first neighbor
    FindFirstAccessNodesOfParent(*first_neighbor, FirstRefs);
  }
}

void TreeStructureQueryEngine::FindLastAccessNodesOfParent(SPSTNode *Node,
                                                           SmallVector<SPSTNode *, 8> &LastRefs)
{
  if (Node->neighbors.empty() && !dynamic_cast<RefTNode *>(Node)) {
    LLVM_DEBUG(dbgs() << "Parent node does not have any child node\n");
    return;
  }
  // if the node is a loop node, we analyze its last neighbor
  // it can be:
  // 1) the last access node of its last subloop
  // 2) the last access node of all branches
  // 3) the last access node
  auto last_neighbor = Node->neighbors.rbegin();
  if (LoopTNode *L = dynamic_cast<LoopTNode *>(Node)) {
    if (LoopTNode *LastSubL = dynamic_cast<LoopTNode *>(*last_neighbor)) {
      FindLastAccessNodesOfParent(LastSubL, LastRefs);
    } else if (RefTNode *LastRef = dynamic_cast<RefTNode *>(*last_neighbor)) {
      LastRefs.push_back(LastRef);
    } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(*last_neighbor)) {
      FindLastAccessNodesOfParent(Branch->neighbors[0], LastRefs);
      // we add if condition to avoid empty false branch
      if (!Branch->neighbors[1]->neighbors.empty())
        FindLastAccessNodesOfParent(Branch->neighbors[1], LastRefs);
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Node)) {
    // the dummy node can be one of the branch condition
    // its last access is the last access of its last neighbor
    FindLastAccessNodesOfParent(*last_neighbor, LastRefs);
  } else if (RefTNode *Ref = dynamic_cast<RefTNode *>(Node)) {
    // this can be reached if and only if
    // 1) check one branch condition
    // 2) root
    LastRefs.push_back(Ref);
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Node)) {
    FindLastAccessNodesOfParent(Branch->neighbors[0], LastRefs);
    if (!Branch->neighbors[1]->neighbors.empty())
      FindLastAccessNodesOfParent(Branch->neighbors[1], LastRefs);
  }
}

void TreeStructureQueryEngine::GetPath(SPSTNode *Src, SPSTNode *Sink,
                                       AccPath &Path, bool ExcludeSink)
{
  SPSTNode *tmpNode = Src;
  // if Sink is NULL, it means finds all path to the root (root does not have
  // an entry in the ParentNodeMapping)
  while (ParentNodeMapping.find(tmpNode) != ParentNodeMapping.end()) {
    if (ParentNodeMapping[tmpNode] != Sink) {
//      LLVM_DEBUG(dbgs() << ParentNodeMapping[tmpNode]->getLoopStringExpr() << " -- ");
      Path.push_back(ParentNodeMapping[tmpNode]);
      tmpNode = ParentNodeMapping[tmpNode];
    } else {
      break;
    }
  }
  if (!ExcludeSink && dynamic_cast<LoopTNode *>(Sink)) {
    Path.push_back(Sink);
//    LLVM_DEBUG(dbgs() << dynamic_cast<LoopTNode *>(Sink)->getLoopStringExpr() << "\n ");
  }
}

void TreeStructureQueryEngine::BuildParentMapping(SPSTNode *Root)
{
  if (LoopTNode *L = dynamic_cast<LoopTNode *>(Root)) {
    auto neighbor_iter = L->neighbors.begin();
    for (; neighbor_iter != L->neighbors.end(); neighbor_iter++) {
      if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(*neighbor_iter)) {
        // we do not care branches, when meeting branch node, we track all nodes
        // in its two branches
        ParentNodeMapping[Branch] = L;
        BuildParentMapping(Branch);
      } else {
        ParentNodeMapping[*neighbor_iter] = L;
        BuildParentMapping(*neighbor_iter);
      }
    }
  } else if (RefTNode *Ref = dynamic_cast<RefTNode *>(Root)) {
    // when reaching here, its parent mapping has already bee build, or it is
    // an access outside a loop, hence has no parent.
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Root)) {
    // track all nodes in its two branches
    auto neighbor_iter = Branch->neighbors[0]->neighbors.begin();
    for (; neighbor_iter != Branch->neighbors[0]->neighbors.end(); neighbor_iter++) {
      // if Branch is not enclosed in any loop, we do not track its parent
      if (ParentNodeMapping.find(Branch) != ParentNodeMapping.end()) {
        ParentNodeMapping[*neighbor_iter] = ParentNodeMapping[Branch];
      }
      BuildParentMapping(*neighbor_iter);
    }

    // we add if condition to avoid empty false branch
    if (!Branch->neighbors[1]->neighbors.empty()) {
      neighbor_iter = Branch->neighbors[1]->neighbors.begin();
      for (; neighbor_iter != Branch->neighbors[1]->neighbors.end();
           neighbor_iter++) {
        if (ParentNodeMapping.find(Branch) != ParentNodeMapping.end()) {
          ParentNodeMapping[*neighbor_iter] = ParentNodeMapping[Branch];
        }
        BuildParentMapping(*neighbor_iter);
      }
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Root)) {
    // the dummy node can be
    // 1) the root of the tree
    // 2) one of the branch condition
    // either case, its first access is the first acceess of its first neighbor
    auto neighbor_iter = Dummy->neighbors.begin();
    for (; neighbor_iter != Dummy->neighbors.end(); neighbor_iter++) {
      BuildParentMapping(*neighbor_iter);
    }
  }
}

bool TreeStructureQueryEngine::areAccessToSameArray(RefTNode *A, RefTNode *B)
{
  if (A->getBase() == B->getBase()) {
    return true;
  } else {
    // check whether these two base has the same name
    return A->getArrayNameString() == B->getArrayNameString();
  }
}

bool TreeStructureQueryEngine::isTopologicallyLargerThan(SPSTNode *A, SPSTNode *B)
{
  assert(A != B && "Cannot distinguish the topology order if A == B");
  stack<SPSTNode *> q;
  q.push(Root);
  bool VisitA = false, VisitB = false;
  while (!q.empty()) {
    SPSTNode *top = q.top();
    q.pop();
    VisitA = (A == top);
    VisitB = (B == top);
    if (VisitA && !VisitB) {
      // A is visited before B
      return true;
    }
    if (!VisitA && VisitB) {
      // A is visited after B
      return false;
    }
    if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(top)) {
      if (!Branch->neighbors[1]->neighbors.empty()) {
        auto neighbor_iter = Branch->neighbors[1]->neighbors.begin();
        for (; neighbor_iter != Branch->neighbors[1]->neighbors.end(); neighbor_iter++) {
          q.push(*neighbor_iter);
        }
      }
      if (!Branch->neighbors[1]->neighbors.empty()) {
        auto neighbor_iter = Branch->neighbors[0]->neighbors.begin();
        for (; neighbor_iter != Branch->neighbors[0]->neighbors.end();
             neighbor_iter++) {
          q.push(*neighbor_iter);
        }
      }
    } else {
      auto neighbor_iter = top->neighbors.rbegin();
      for (; neighbor_iter != top->neighbors.rend(); neighbor_iter++) {
        q.push(*neighbor_iter);
      }
    }
  }
  return false;
}

// when searching reuse of Target access, do we need to check Node
bool TreeStructureQueryEngine::isReachable(SPSTNode *Node, RefTNode *Target)
{
  bool ret = false;
  if (RefTNode *Ref = dynamic_cast<RefTNode *>(Node)) {
    // Target is Reachable from Ref if:
    // 1. They are in the same loop nest (has common loop dominator)
    // 2. Target is ahead of Ref (Target is traversed first then Target when we
    // scan the tree level-by-level
    SmallVector<SPSTNode *, 8> Dominators;
    GetImmediateCommonLoopDominator(Ref, Target, false, Dominators);
    ret = !Dominators.empty() || isTopologicallyLargerThan(Target, Ref);
    if (ret) {
      ret &= areAccessToSameArray(Ref, Target);
    }
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Node)) {
    if (!Branch->neighbors[0]->neighbors.empty()) {
      ret |= isReachable(Branch->neighbors[0], Target);
    }
    if (!Branch->neighbors[1]->neighbors.empty()) {
      ret |= isReachable(Branch->neighbors[1], Target);
    }
    return ret;
  } else if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(Node)) {
    // 1. Loop dominates Target or the parent of the Target
    // 2. Loop or its neighbor contains the node that visit the same address as
    // Target and Loop is the loop visited after Target is visited
    AccPath path;
    GetPath(Target, Loop, path, false);
    ret = !path.empty() || isTopologicallyLargerThan(Target, Loop);
    if (ret) {
      ret = false;
      // find if any of its appended refnode access the same array as the Target
      vector<RefTNode *> RefsInLoop;
      FindAllRefsInLoop(Loop, RefsInLoop);
      for (auto ref : RefsInLoop) {
        if (isReachable(ref, Target)) {
          ret = true;
          break;
        }
      }
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Node)) {
    // at least one of its neighbor isReachable to Target
    auto neighbor_iter = Dummy->neighbors.begin();
    for (;neighbor_iter != Dummy->neighbors.end(); neighbor_iter++) {
      if (isReachable(*neighbor_iter, Target)) {
        ret = true;
        break;
      }
    }
  }
  return ret;
}

bool TreeStructureQueryEngine::hasBranchInside(LoopTNode *L)
{
  for (auto neighbor : L->neighbors) {
    if (dynamic_cast<RefTNode *>(neighbor)) {
      return false;
    } else if (LoopTNode *SubL = dynamic_cast<LoopTNode *>(neighbor)) {
      return hasBranchInside(SubL);
    } else if (dynamic_cast<BranchTNode *>(neighbor)) {
      return true;
    }
  }
  return false;
}

bool TreeStructureQueryEngine::hasConstantLoopBound(LoopTNode *LoopNode)
{
  LoopBound *LB = LoopNode->getLoopBound();
  if (LB && LB->FinalValue) {
    bool isLBConstant = isa<ConstantInt>(LB->InitValue);
    bool isUBConstant = isa<ConstantInt>(LB->FinalValue);
    return (isLBConstant && isUBConstant);
  }
  return false;
}

bool TreeStructureQueryEngine::hasConstantLoopLowerBound(LoopTNode *LoopNode)
{
  LoopBound *LB = LoopNode->getLoopBound();
  if (LB && LB->FinalValue) {
    bool isLBConstant = isa<ConstantInt>(LB->InitValue);
    return isLBConstant;
  }
  return false;
}

bool TreeStructureQueryEngine::hasConstantLoopUpperBound(LoopTNode *LoopNode)
{
  LoopBound *LB = LoopNode->getLoopBound();
  if (LB && LB->FinalValue) {
    bool isUBConstant = isa<ConstantInt>(LB->FinalValue);
    return isUBConstant;
  }
  return false;
}

void TreeStructureQueryEngine::GetSubLoopTNode(SPSTNode *Node,
                                               SmallVectorImpl<LoopTNode *> &Childs)
{
  for (auto neighbor : Node->neighbors) {
    if (LoopTNode *SubL = dynamic_cast<LoopTNode *>(neighbor)) {
      Childs.push_back(SubL);
      GetSubLoopTNode(SubL, Childs);
    } else if (dynamic_cast<BranchTNode *>(neighbor)) {
      GetSubLoopTNode(neighbor->neighbors[0], Childs);
      if (!neighbor->neighbors[0]->neighbors.empty())
        GetSubLoopTNode(neighbor->neighbors[1], Childs);
    } else if (dynamic_cast<DummyTNode *>(neighbor)) {
      for (auto node : neighbor->neighbors) {
        GetSubLoopTNode(node, Childs);
      }
    }
  }
}

void TreeStructureQueryEngine::FindAllRefsInLoop(SPSTNode *Node,
                                                 vector<RefTNode *> &Refs)
{
  if (RefTNode *Ref = dynamic_cast<RefTNode *>(Node)) {
    Refs.push_back(Ref);
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Node)) {
    FindAllRefsInLoop(Branch->neighbors[0], Refs);
    if (!Branch->neighbors[1]->neighbors.empty()) {
      FindAllRefsInLoop(Branch->neighbors[1], Refs);
    }
  } else {
    auto neighbor_iter = Node->neighbors.begin();
    for (;neighbor_iter != Node->neighbors.end(); neighbor_iter++) {
      FindAllRefsInLoop(*neighbor_iter, Refs);
    }
  }
}


string GetSPSTNodeName(SPSTNode *Node)
{
  if (RefTNode *Ref = dynamic_cast<RefTNode *>(Node)) {
    return Ref->getRefExprString();
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Node)) {
    return Branch->getConditionExpr();
  } else if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(Node)) {
    return Loop->getLoopStringExpr();
  }
  return Node->getName();
}

