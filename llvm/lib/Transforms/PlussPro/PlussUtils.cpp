//
// Created by noya-fangzhou on 11/5/21.
//

#include <regex>
#include "PlussUtils.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "math.h"
#define DEBUG_TYPE  "pluss-utils"

void FindAllPathesBetweenTwoBlockImpl(BasicBlock *Src, BasicBlock *Sink,
                                      unordered_map<BasicBlock *, bool> &visited,
                                      Path &path, SmallVectorImpl<Path> &Pathes);

string StringTranslator::ValueToStringExpr(Value *V, TranslateStatus &status)
{
  if (!V) {
    status = NOT_TRANSLATEABLE;
    return "";
  }
  string expr = "";
  if (isa<ConstantInt>(V)) {
    ConstantInt *ConstI = dyn_cast<ConstantInt>(V);
    return to_string(ConstI->getValue().getSExtValue());
  } else if (isa<ConstantFP>(V)) {
    ConstantFP *ConstFP = dyn_cast<ConstantFP>(V);
    return to_string(ConstFP->getValue().convertToDouble());
  } else if (isa<Argument>(V)) {
    Argument *Arg = dyn_cast<Argument>(V);
    return Arg->getName().str();
  }
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) {
    if (V->hasName()) {
      return V->getName().str();
    } else if (isa<IntrinsicInst>(V)) {
      string expr = "";
      IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V);
      switch (Intrinsic->getIntrinsicID()) {
      case Intrinsic::exp:
        expr += "exp(";
        break;
      case Intrinsic::sin:
        expr += "sin(";
        break;
      case Intrinsic::cos:
        expr += "cos(";
        break;
      case Intrinsic::ceil:
        expr += "ceil(";
        break;
      case Intrinsic::floor:
        expr += "floor(";
        break;
      case Intrinsic::pow:
        expr += "pow(";
        break;
      default:
        status = NOT_TRANSLATEABLE;
        break;
      }
      if (status == SUCCESS) {
        for (unsigned i = 0; i < Intrinsic->getNumArgOperands(); i++) {
          expr += ValueToStringExpr(Intrinsic->getArgOperand(i), status);
          if (status != SUCCESS)
            break;
          if (i < Intrinsic->getNumArgOperands()-1)
            expr += ", ";
        }
        if (status == SUCCESS)
          expr += ")";
      }
      return expr;
    } else {
      status = NOT_TRANSLATEABLE;
      return "";
    }
  }
  switch(I->getOpcode()) {
  case Instruction::FAdd:
  case Instruction::Add: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " + " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::FSub:
  case Instruction::Sub: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " - " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::FMul:
  case Instruction::Mul: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " * " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " / " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::LShr: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " << " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::Xor: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " ^ " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::And: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " & " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::Or: {
    expr = "(" + ValueToStringExpr(I->getOperand(0), status)
           + " | " + ValueToStringExpr(I->getOperand(1), status) + ")";
    break;
  }
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::SExt:
  case Instruction::ZExt: {
    expr = ValueToStringExpr(I->getOperand(0), status);
    break;
  }
  case Instruction::Load: {
    expr = I->getOperand(0)->getName().str();
    break;
  }
  case Instruction::PHI: {
    PHINode *Phi = dyn_cast<PHINode>(I);
    if (InductionVarNameTable.find(Phi) != InductionVarNameTable.end()) {
      expr = InductionVarNameTable.at(Phi);
    } else if (Phi->getNumIncomingValues() == 1) {
      // in lcssa form, the array subscript could also be represented by phi
      // node with only one branches, in this case, we have to pass its operand
      // to ValueToStringExpr()
      // for example:
      // i64 %idxprom33, i64 %idxprom35
      // %idxprom33.lcssa = phi i64 [ %idxprom33, %for.cond29 ]
      expr = ValueToStringExpr(Phi->getOperand(0), status);
    } else {
      expr = V->getName().str();
    }
    break;
  }
  case Instruction::Alloca: {
    expr = V->getName().str();
    break;
  }
  case Instruction::FCmp:
  case Instruction::ICmp: {
    CmpInst *CI = dyn_cast<CmpInst>(I);
    expr = ConditionToStringExpr(CI, status);
    break;
  }
  case Instruction::Call: {
    CallInst *FuncCall = dyn_cast<CallInst>(I);
    LLVM_DEBUG(
        dbgs() << *FuncCall << " has " << FuncCall->getNumArgOperands() << " arguments\n";
//          for (unsigned i = 0; i < FuncCall->getNumArgOperands(); i++) {
//            string arg = ValueToStringExpr(FuncCall->getArgOperand(i), status);
//            if (status == SUCCESS) {
//              dbgs() << "args[" << i << "] = " << arg;
//              if (i < FuncCall->getNumArgOperands() - 1)
//                dbgs() << ", ";
//            }
//          }
    );

    break;
  }
  default:
    status = NOT_TRANSLATEABLE;
    break;
  }
  return expr;
}

string StringTranslator::ConditionToStringExpr(CmpInst *CI,
                                               TranslateStatus &status,
                                               bool isDotFormat)
{
  string Expr = "";
  Expr += ValueToStringExpr(CI->getOperand(0), status);
  if (status != SUCCESS)
    return Expr;
  string sign = PredicateToStringExpr(CI->getPredicate(), status);
  if (isDotFormat)
    sign = "\\" + sign;
  if (status != SUCCESS)
    return Expr;
  Expr += (" " + sign + " ");
  Expr += ValueToStringExpr(CI->getOperand(1), status);
  return Expr;
}

string StringTranslator::PredicateToStringExpr(CmpInst::Predicate P,
                                               TranslateStatus &status)
{
  string sign = "";
  switch(P) {
  case llvm::CmpInst::ICMP_EQ:
  case llvm::CmpInst::FCMP_OEQ:
  case llvm::CmpInst::FCMP_UEQ:
    sign = "==";
    break;
  case llvm::CmpInst::ICMP_NE:
  case llvm::CmpInst::FCMP_ONE:
  case llvm::CmpInst::FCMP_UNE:
    sign = "!=";
    break;
  case llvm::CmpInst::ICMP_SGT:
  case llvm::CmpInst::ICMP_UGT:
  case llvm::CmpInst::FCMP_OGT:
  case llvm::CmpInst::FCMP_UGT:
    sign = ">";
    break;
  case llvm::CmpInst::ICMP_SGE:
  case llvm::CmpInst::ICMP_UGE:
  case llvm::CmpInst::FCMP_OGE:
  case llvm::CmpInst::FCMP_UGE:
    sign = ">=";
    break;
  case llvm::CmpInst::ICMP_SLT:
  case llvm::CmpInst::ICMP_ULT:
  case llvm::CmpInst::FCMP_OLT:
  case llvm::CmpInst::FCMP_ULT:
    sign = "<";
    break;
  case llvm::CmpInst::ICMP_SLE:
  case llvm::CmpInst::ICMP_ULE:
  case llvm::CmpInst::FCMP_OLE:
  case llvm::CmpInst::FCMP_ULE:
    sign = "<=";
    break;
  default:
    status = NOT_TRANSLATEABLE;
    break;
  }
  return sign;
}

string StringTranslator::SCEVToStringExpr(const SCEV *S, Loop *L,
                                          TranslateStatus &status)
{
  string ret;
  switch (S->getSCEVType()) {
  case scAddExpr: {
    ret = "(";
    const SCEVAddExpr *Mul = cast<SCEVAddExpr>(S);
    for (unsigned i = 0; i < Mul->getNumOperands(); i++) {
      ret += SCEVToStringExpr(Mul->getOperand(i), L, status);
      if (status != SUCCESS)
        break;
      if (i != Mul->getNumOperands() - 1)
        ret += " * ";
    }
    ret += ")";
    break;
  }
  case scMulExpr: {
    ret = "(";
    const SCEVMulExpr *Mul = cast<SCEVMulExpr>(S);
    for (unsigned i = 0; i < Mul->getNumOperands(); i++) {
      ret += SCEVToStringExpr(Mul->getOperand(i), L, status);
      if (status != SUCCESS)
        break;
      if (i != Mul->getNumOperands() - 1)
        ret += " * ";
    }
    ret += ")";
    break;
  }
  case scUDivExpr: {
    ret = "(";
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(S);
    ret += (SCEVToStringExpr(UDiv->getLHS(), L, status) + " / " +
            SCEVToStringExpr(UDiv->getRHS(), L, status));
    ret += ")";
    break;
  }
  case scSignExtend:
  case scTruncate:
  case scZeroExtend: {
    const SCEVCastExpr *CastS = cast<SCEVCastExpr>(S);
    ret = SCEVToStringExpr(CastS->getOperand(), L, status);
    break;
  }
  case scSMinExpr:
  case scUMinExpr: {
    const SCEVMinMaxExpr *MinMaxS = cast<SCEVMinMaxExpr>(S);
    ret = "min(";
    for (unsigned i = 0; i < MinMaxS->getNumOperands(); i++) {
      ret += SCEVToStringExpr(MinMaxS->getOperand(i), L, status);
      if (status != SUCCESS)
        break;
      if (i != MinMaxS->getNumOperands()-1)
        ret += ", ";
    }
    ret += ")";
  }
    break;
  case scSMaxExpr:
  case scUMaxExpr: {
    const SCEVMinMaxExpr *MinMaxS = cast<SCEVMinMaxExpr>(S);
    ret = "max(";
    for (unsigned i = 0; i < MinMaxS->getNumOperands(); i++) {
      ret += SCEVToStringExpr(MinMaxS->getOperand(i), L, status);
      if (status != SUCCESS)
        break;
      if (i != MinMaxS->getNumOperands()-1)
        ret += ", ";
    }
    ret += ")";
    break;
  }
  case scAddRecExpr: {
    // could be loop induction variable
    const SCEVAddRecExpr *AddRecS = cast<SCEVAddRecExpr>(S);
    LLVM_DEBUG(dbgs() << *AddRecS
                      << "Start: " << *(AddRecS->getStart()) << " "
                      << "Recurr:"
                      << *(AddRecS->getStepRecurrence(*(this->SE)))
                      << "\n");
    if (AddRecS->getLoop()->getCanonicalInductionVariable()) {
        LLVM_DEBUG(
          dbgs() << *(AddRecS->getLoop()->getCanonicalInductionVariable()) << "\n");
        ret = ValueToStringExpr(AddRecS->getLoop()->getCanonicalInductionVariable(),
                                status);
    } else {
      status = NOT_TRANSLATEABLE;
    }
    break;
  }
  case scConstant: {
    const SCEVConstant *ConstS = cast<SCEVConstant>(S);
    ret = to_string(ConstS->getAPInt().getSExtValue());
    break;
  }
  case scUnknown: {
    const SCEVUnknown * UnknownS = cast<SCEVUnknown>(S);
    // Here is a trick.
    // If one of the operand in this SCEV is loop invariant but we do
    // not know its value during the compile time. It is okay.
    // For example, this SCEC represents alpha * i + c, alpha and c is unknown
    // but they are loop invariant. In this case, when we check whether they
    // touch the same address two iteration later, say, at i == 1, the address
    // would be alpha + c, and alpha * 2 + c at i == 2. In this case, alpha + c
    // == alpha * 2 + c if and only if alpha == 0;
    ret = ValueToStringExpr(UnknownS->getValue(), status);
    if (!L->isLoopInvariant(UnknownS->getValue()))
      status = NOT_TRANSLATEABLE;
    break;
  }
  default:
    status = NOT_TRANSLATEABLE;
    break;
  }
  return ret;
}

bool InductionVariableArithmeticInst(Value *V,
                                            SmallPtrSetImpl<PHINode *> &InductionPHIs)
{
  Constant *C = cast<Constant>(V);
  if (C)
    return true;
  Instruction *I = cast<Instruction>(V);
  if (!I)
    return false;
  bool ret = true;
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Mul:
  case Instruction::FDiv:
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Or:
  case Instruction::And:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::AShr:{
    for (unsigned i = 0; i < I->getNumOperands(); i++) {
      if (!InductionVariableArithmeticInst(I->getOperand(i), InductionPHIs)) {
        ret = false;
        break;
      }
    }
    break;
  }
  case Instruction::PHI: {
    PHINode *Phi = cast<PHINode>(V);
    ret = (InductionPHIs.find(Phi) != InductionPHIs.end());
    break;
  }
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::FPExt:
  case Instruction::Trunc: {
    ret = InductionVariableArithmeticInst(I->getOperand(0), InductionPHIs);
    break;
  }
  default:
    ret = false;
    break;
  }
  return ret;
}

/// BFS of all successors of Target
/// The immediate post dominator of the Target block is its
/// first successor that post dominates it.
BasicBlock *ImmediatePostDominator(PostDominatorTree &PDT, BasicBlock *Target)
{
  queue<BasicBlock *>queue;
  unordered_set<BasicBlock *>visit;
  queue.push(Target);
  while (!queue.empty()) {
    BasicBlock *Head = queue.front();
    queue.pop();
    if (visit.find(Head) != visit.end())
      continue;
    visit.insert(Head);
    if (Head != Target && PDT.dominates(Head, Target))
      return Head;
    auto blockIter = succ_begin(Head);
    while (blockIter != succ_end(Head)) {
      if (visit.find(*blockIter) == visit.end())
        queue.push(*blockIter);
      blockIter++;
    }
  }
  return nullptr;
}


/// do Depth First Traversal of given directed graph.
/// Start the traversal from source. Keep storing the visited vertices
/// in an unordere_set say ‘visit[]’. If we reach the destination vertex,
/// print contents of visit[]. The important thing is to mark current vertices
/// in visit[] as beingVisited, so that the traversal doesn’t go in a cycle.
void FindAllPathesBetweenTwoBlock(BasicBlock *Src, BasicBlock *Sink,
                                  SmallVectorImpl<Path> &Pathes)
{
  Path path;
  queue<Path > q;
  path.push_back(Src);
  q.push(path);

  while (!q.empty()) {
    Path temp = q.front();
    q.pop();
    BasicBlock *Back = temp.back();
    if (Back == Sink) {
      LLVM_DEBUG(
      for (auto b : temp) {
        dbgs() << b->getName() << " -> ";
      }
          dbgs() << "\n";
      );
      Pathes.push_back(temp);
    }
    auto SuccIter = succ_begin(Back);
    while (SuccIter != succ_end(Back)) {
//      LLVM_DEBUG(
//          dbgs() << (*SuccIter)->getName() << " is a successor of "
//                        << Src->getName() << " \n";
//          if (find(temp.begin(), temp.end(), *SuccIter) != temp.end())
//              dbgs() << (*SuccIter)->getName() << " has been touched\n";);
      if (find(temp.begin(), temp.end(), *SuccIter) == temp.end()) {
        Path newpath(temp);
        newpath.push_back(*SuccIter);
        q.push(newpath);
      }
      SuccIter++;
    }
  }
}

/*
// utility function for printing
// the found path in graph
void printpath(vector& path)
{
  int size = path.size();
  for (int i = 0; i < size; i++)
    cout << path[i] << " ";
  cout << endl;
}

// utility function to check if current
// vertex is already present in path
int isNotVisited(int x, vector& path)
{
  int size = path.size();
  for (int i = 0; i < size; i++)
    if (path[i] == x)
      return 0;
  return 1;
}

// utility function for finding paths in graph
// from source to destination
void findpaths(vector<vector >&g, int src,
               int dst, int v)
{
  // create a queue which stores
  // the paths
  queue<vector > q;

  // path vector to store the current path
  vector path;
  path.push_back(src);
  q.push(path);
  while (!q.empty()) {
    path = q.front();
    q.pop();
    int last = path[path.size() - 1];

    // if last vertex is the desired destination
    // then print the path
    if (last == dst)
      printpath(path);

    // traverse to all the nodes connected to
    // current vertex and push new path to queue
    for (int i = 0; i < g[last].size(); i++) {
      if (isNotVisited(g[last][i], path)) {
        vector newpath(path);
        newpath.push_back(g[last][i]);
        q.push(newpath);
      }
    }
  }
}
*/
void FindAllPathesBetweenTwoBlockImpl(BasicBlock *Src, BasicBlock *Sink,
                                      unordered_map<BasicBlock *, bool> &visited,
                                      Path &path, SmallVectorImpl<Path> &Pathes)
{
  visited[Src] = true;
  path.push_back(Src);

  auto SuccIter = succ_begin(Src);
  while (SuccIter != succ_end(Src)) {
    LLVM_DEBUG(dbgs() << (*SuccIter)->getName() << " is a successor of "
                      << Src->getName() << " \n";
               if (visited[*SuccIter])
                   dbgs() << (*SuccIter)->getName() << " has been touched\n";);
    if (visited.find(*SuccIter) == visited.end() || !visited[*SuccIter]) {
      if (*SuccIter != Sink) {
        FindAllPathesBetweenTwoBlockImpl(*SuccIter, Sink, visited, path,
                                         Pathes);
      } else {
        path.push_back(*SuccIter);
        LLVM_DEBUG(
            for (auto BB: path) {
              dbgs() << BB->getName() << " -> ";
            }
            dbgs() << "\n";
        );
      }
    }
    SuccIter++;
  }
  //remove from path
  visited[Src] = false;
}


bool FindValueInInstruction(Value *Candidate, Value *Target)
{
  if (Candidate == Target)
    return true;
  Instruction *I = dyn_cast<Instruction>(Candidate);
  // Current we assumes Target is in PHINode type
  if (!I)
    return false;
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
  case Instruction::Or: {
    return (FindValueInInstruction(I->getOperand(0), Target) ||
        FindValueInInstruction(I->getOperand(0), Target));
  }
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::SExt:
  case Instruction::ZExt: {
    return FindValueInInstruction(I->getOperand(0), Target);
  }
  case Instruction::PHI: {
    if (!isa<PHINode>(Target))
      return false;
    PHINode *CandidatePhi = dyn_cast<PHINode>(I);
    PHINode *TargetPhi = dyn_cast<PHINode>(Target);
    return TargetPhi == CandidatePhi;
  }
  default:
    break;
  }
  return false;
}


bool FindValueInSCEV(const SCEV *S, Value *Target, ScalarEvolution &SE)
{
  // Current we assumes Target is in PHINode type
  if (!S)
    return false;

  bool ret = false;
  switch (S->getSCEVType()) {
  case scAddExpr:
  case scMulExpr: {
    const SCEVNAryExpr *NAryS = cast<SCEVNAryExpr>(S);
    for (unsigned i = 0; i < NAryS->getNumOperands(); i++) {
      ret |= FindValueInSCEV(NAryS->getOperand(i), Target, SE);
    }
    break;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(S);
    ret = (FindValueInSCEV(UDiv->getLHS(), Target, SE) ||
        FindValueInSCEV(UDiv->getRHS(), Target, SE));
    break;
  }
  case scSignExtend:
  case scTruncate:
  case scZeroExtend: {
    const SCEVCastExpr *CastS = cast<SCEVCastExpr>(S);
    ret = FindValueInSCEV(CastS->getOperand(), Target, SE);
    break;
  }
  case scSMinExpr:
  case scUMinExpr:
  case scSMaxExpr:
  case scUMaxExpr: {
    const SCEVMinMaxExpr *MinMaxS = cast<SCEVMinMaxExpr>(S);
    for (unsigned i = 0; i < MinMaxS->getNumOperands(); i++) {
      ret |= FindValueInSCEV(MinMaxS->getOperand(i), Target, SE);
    }
    break;
  }
  case scAddRecExpr: {
    // could be loop induction variable
    const SCEVAddRecExpr *AddRecS = cast<SCEVAddRecExpr>(S);
//    PHINode *InductionPhi = getInductionVariable(AddRecS->getLoop(), SE);
//    return FindValueInInstruction(InductionPhi, Target);
    break;
  }
  case scConstant: {
    const SCEVConstant *ConstS = cast<SCEVConstant>(S);
    if (isa<ConstantInt>(Target) && ConstS->getType()->isIntegerTy()) {
      ConstantInt *ConstTarget = dyn_cast<ConstantInt>(Target);
      ret = (ConstTarget->getSExtValue() == ConstS->getAPInt().getSExtValue());
    }
    break;
  }
  case scUnknown: {
    const SCEVUnknown * UnknownS = cast<SCEVUnknown>(S);
    return FindValueInInstruction(UnknownS->getValue(), Target);
  }
  default:
    break;
  }
  return ret;
}

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
    LLVM_DEBUG(dbgs() << *PhiVar << "\n");
    if (!SE.isSCEVable(PhiTy))
      continue;
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

int ReplaceValueWithConstant(Value *V, unordered_map<Value *, int> & constants, TranslateStatus &status)
{
  if (!V) {
    status = FAIL;
    return -1;
  }
  int retVal = 0;
  if (isa<ConstantInt>(V)) {
    ConstantInt *ConstI = dyn_cast<ConstantInt>(V);
    return ConstI->getValue().getSExtValue();
  }
  Instruction *I = dyn_cast<Instruction>(V);
#if 0
  if (!I) {
    if (isa<IntrinsicInst>(V)) {
      string expr = "";
      IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V);
      switch (Intrinsic->getIntrinsicID()) {
      case Intrinsic::exp:
        retVal = exp(ReplaceValueWithConstant(Intrinsic->getOperand(0), constant, status));
        break;
      case Intrinsic::sin:
        retVal = sin(ReplaceValueWithConstant(Intrinsic->getOperand(0), constant, status));
        break;
      case Intrinsic::cos:
        retVal = sin(ReplaceValueWithConstant(Intrinsic->getOperand(0), constant, status));
        break;
      case Intrinsic::ceil:
        retVal = ceil(ReplaceValueWithConstant(Intrinsic->getOperand(0), constant, status));
        break;
      case Intrinsic::floor:
        retVal = floor(ReplaceValueWithConstant(Intrinsic->getOperand(0), constant, status));
        break;
      case Intrinsic::pow:
        retVal = pow(ReplaceValueWithConstant(Intrinsic->getOperand(0), constant, status));
        break;
      default:
        status = NOT_TRANSLATEABLE;
        break;
      }
      return retVal;
    } else {
      status = NOT_TRANSLATEABLE;
      return -1;
    }
  }
#endif
  switch(I->getOpcode()) {
  case Instruction::FAdd:
  case Instruction::Add: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) +
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::FSub:
  case Instruction::Sub: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) -
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::FMul:
  case Instruction::Mul: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) *
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) /
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::LShr: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) <<
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::Xor: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) ^
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::And: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) &
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::Or: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status) |
             ReplaceValueWithConstant(I->getOperand(1), constants, status);
    break;
  }
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::SExt:
  case Instruction::ZExt: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status);
    break;
  }
  case Instruction::Load: {
    retVal = ReplaceValueWithConstant(I->getOperand(0), constants, status);
    break;
  }
  case Instruction::PHI: {
    retVal = constants[V];
    break;
  }
  default:
    status = FAIL;
    break;
  }
  if (status == SUCCESS)
    return retVal;
  return -1;
}

// replace all 'del_str' with 'new_str" in base
void ReplaceSubstringWith(string &base, string del_str, string new_str)
{
  size_t index = 0;
  while (true) {
    /* Locate the substring to replace. */
    index = base.find(del_str, index);
    if (index == std::string::npos)
      break;
    /* Make the replacement. */
    base.replace(index, del_str.size(), new_str);

    /* Advance index forward so the next iteration doesn't pick it up as well. */
    index += new_str.size();
  }
}

bool isConstantString(string s)
{
  return regex_match(s, regex("^\\d+"));
}
