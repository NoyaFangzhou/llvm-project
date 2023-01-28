//
// Created by noya-fangzhou on 7/15/22.
// CRI Model w/ Island-hopping OPT sampler codegen
//

#include "ModelCodeGenPro.h"
#include "AccessGraphAnalysis.h"
#include "InductionVarAnalysis.h"
#include "LoopAnalysisWrapperPass.h"
#include "PlussAbstractionTreeAnalysis.h"
#include "ModelValidationAnalysis.h"
#include "ReferenceAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils.h"
#include <algorithm>

#define DEBUG_TYPE "new-model-codegen"

using namespace std;
using namespace llvm;

namespace ModelCodeGenPro {

char ModelCodeGenProWrapperPass::ID = 0;
static RegisterPass<ModelCodeGenProWrapperPass>
    X("new-model-codegen", "Pass that generate the model code w/ Island-hopping optimization");

ModelCodeGenProWrapperPass::ModelCodeGenProWrapperPass() : FunctionPass(ID) {}

// the trip is represented as the (ub - lb + 1) / step if <=/>= or
// (ub - lb) / step if </>
// default, the trip is (ub - lb), assuming < and step is 1
string ModelCodeGenProWrapperPass::RepresentLoopTrip(LoopTNode *LN)
{
  string trip = "";
  TranslateStatus status = SUCCESS;
  string induction_expr = Translator->ValueToStringExpr(LN->getInductionPhi(), status);
  // default one
  trip = "(" + induction_expr + "_ub - " + induction_expr + "_lb";

  string loop_init_expr = Translator->ValueToStringExpr(LN->getLoopBound()->InitValue, status),
      loop_final_expr = Translator->ValueToStringExpr(LN->getLoopBound()->FinalValue, status);

  if (status == SUCCESS)
    trip = "(" + loop_final_expr + "-" + loop_init_expr;

  switch(LN->getLoopBound()->Predicate) {
  case CmpInst::ICMP_ULE:
  case CmpInst::ICMP_SLE:
  case CmpInst::ICMP_UGE:
  case CmpInst::ICMP_SGE: {
    trip + " + 1";
  }
    break;
  default:
    break;
  }
  trip += ")";
  string loop_step = ReplaceInductionVarExprByType(
      LN->getLoopBound()->StepValue, SAMPLE, status);
  if (loop_step != "1")
    trip = "(" + trip + " / " + loop_step + ")";
  return trip;
}

int ModelCodeGenProWrapperPass::ComputeDistanceToParallelLoop(LoopTNode *LN)
{
  int distance = 0; // assume LN is the parallel loop
  bool FindParallelLoop = false;
  // phi is either an induction variable inside a parallel loop or it
  // is an induction variable of a plain loop
  if (LN) {
    // if the phi is the induction of the parallel loop
    if (LN->isParallelLoop())
      return 0;
    // now we backtrack the parent path from the TargetLoopNode
    LoopTNode *tmp = LN;
    while (ParentMappingInAbstractionTree.find(tmp)
           != ParentMappingInAbstractionTree.end()) {
      if (tmp->isParallelLoop()) {
        FindParallelLoop = true;
        break;
      }
      distance += 1;
      tmp = ParentMappingInAbstractionTree[tmp];
    }
    // if the parallel loop is the top level loop
    // it does not have an entry in ParentMappingInAbstractionTree
    // so we need to check tmp here again
    if (tmp && tmp->isParallelLoop())
      FindParallelLoop = true;
  }

  if (!FindParallelLoop)
    return -1;
  return distance;
}


int ModelCodeGenProWrapperPass::isInsideParallelLoopNest(PHINode *Phi)
{
  LoopTNode *TargetLoopNode = nullptr;
  for (auto loop : LoopNodes) {
    if (loop->getInductionPhi() == Phi) {
      TargetLoopNode = loop;
      break;
    }
  }
  return ComputeDistanceToParallelLoop(TargetLoopNode);
}

ReuseType ModelCodeGenProWrapperPass::isInterThreadSharingRef(RefTNode *RefNode)
{
  // [i][i], [i][j], [j][k]
  unsigned parallel_liv_cnt = 0, parallel_liv_idx = 0;
  ReuseType type = OTHER;
  for (unsigned i = 0; i < RefNode->getSubscripts().size(); i++) {
    Value *Index = RefNode->getSubscripts()[i];
    if (hasParallelLoopInductionVar(Index)) {
      // Parallel Loop IV
      parallel_liv_cnt += 1;
      parallel_liv_idx = i;
    }
  }
  if (parallel_liv_cnt == 0) {
    type = FULL_SHARE;
  } else if (parallel_liv_cnt == 1 && parallel_liv_idx == RefNode->getSubscripts().size()-1) {
    type = SPATIAL_SHARE;
  }
  return type;
}

string ModelCodeGenProWrapperPass::ReplaceInductionVarExprByType(
    Value *V, CodeGenType type, TranslateStatus &status) {
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
          expr += ReplaceInductionVarExprByType(Intrinsic->getArgOperand(i),
                                                type, status);
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
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " + " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::FSub:
  case Instruction::Sub: {
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " - " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::FMul:
  case Instruction::Mul: {
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " * " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv: {
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " / " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::LShr: {
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " << " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::Xor: {
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " ^ " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::And: {
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " & " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::Or: {
    expr = "(" + ReplaceInductionVarExprByType(I->getOperand(0), type, status) + " | " +
           ReplaceInductionVarExprByType(I->getOperand(1), type, status) + ")";
    break;
  }
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::SExt:
  case Instruction::ZExt: {
    expr = ReplaceInductionVarExprByType(I->getOperand(0), type, status);
    break;
  }
  case Instruction::Load: {
    expr = I->getOperand(0)->getName().str();
    break;
  }
  case Instruction::PHI: {
    PHINode *Phi = dyn_cast<PHINode>(I);
    if (InductionVarNameTable.find(Phi) != InductionVarNameTable.end()) {
      // we need to be careful here, Phi may not be the induction variable
      // inside the parallel region, in this case, we need to return its
      // original name inside the InductionVarNameTable.
      string indvar_name = InductionVarNameTable[Phi];

      switch (type) {
      case SAMPLE: {
      }
        break;
      case OMP_PARALLEL:{
        // TODO: check the LoopTNode whose induction phi is Phi
        //  if it is inside a parallel loop nest, we compute its distance 'd' to
        //  the parallel loop and replace its expr as 'progress[tid_to_run].iteration[d]'
        //  else, we return its original name.
        int distance = isInsideParallelLoopNest(Phi);
        if (distance >= 0) {
          indvar_name = "progress[tid_to_run]->iteration[" + to_string(distance) + "]";
        }
      }
        break;
      case OMP_PARALLEL_INIT: {
        int distance = isInsideParallelLoopNest(Phi);
        if (distance >= 0) {
          indvar_name = "parallel_iteration_vector[" + to_string(distance) + "]";
        }
        break;
      }
      case MODEL: {
        for (auto loop : LoopNodes) {
          if (loop->getInductionPhi() == Phi) {
            AccPath  pathToTreeRoot;
            QueryEngine->GetPath(loop, nullptr, pathToTreeRoot);
            if (loop->isParallelLoop())
              indvar_name = "c";
            else
              indvar_name = "src->ivs[" + to_string(pathToTreeRoot.size()) + "]";
            break;
          }
        }
        break;
      }
      default:
        break;
      }
      expr = indvar_name;
    } else if (Phi->getNumIncomingValues() == 1) {
      // in lcssa form, the array subscript could also be represented by phi
      // node with only one branches, in this case, we have to pass its operand
      // to ValueToStringExpr()
      // for example:
      // i64 %idxprom33, i64 %idxprom35
      // %idxprom33.lcssa = phi i64 [ %idxprom33, %for.cond29 ]
      expr = ReplaceInductionVarExprByType(Phi->getOperand(0), type, status);
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
    expr = (ReplaceInductionVarExprByType(CI->getOperand(0), type, status)
            + Translator->PredicateToStringExpr(CI->getPredicate(), status)
            + ReplaceInductionVarExprByType(CI->getOperand(1), type, status));
    break;
  }
  default:
    status = NOT_TRANSLATEABLE;
    break;
  }
  return expr;
}

string ModelCodeGenProWrapperPass::EmitRefNodeAccessExprAtIdx(Value *Index,
                                                              bool isInParallel) {
  TranslateStatus status = SUCCESS;
  // not all indices are inside the parallel loop, for those induction variable
  // outside the parallel loop, we use its original value (ValueToStringExpr)
  if (!isInParallel) {
    Translator->ValueToStringExpr(Index, status);
  } else {
    // need to replace all induction variable name using progress[tid].iteration
    ReplaceInductionVarExprByType(Index, SAMPLE, status);
  }
  if (status != SUCCESS) {
    return "";
  }
  stringstream ss;
  if (!isInParallel) {
    ss << Translator->ValueToStringExpr(Index, status);
  } else {
    ss << ReplaceInductionVarExprByType(Index, OMP_PARALLEL, status);
  }
  return ss.str();
}

vector<string> ModelCodeGenProWrapperPass::EmitRefNodeAccessExpr(RefTNode *RefNode,
                                                                 bool isInParallel)
{
  vector<string> params;
  TranslateStatus status = SUCCESS;
  string base = Translator->ValueToStringExpr(RefNode->getBase(), status);
  // for those accesses that alias, we represent them with a new name
  if (PointerAliasPairs.find(RefNode) != PointerAliasPairs.end()) {
    base = PointerAliasPairs[RefNode];
  }
  if (status == SUCCESS) {
    params.push_back("\"" + base + "\"");
    stringstream ss;
    ss << "{";
    unsigned i = 0;
    for (auto Index : RefNode->getSubscripts()) {
      // not all indices are inside the parallel loop, for those induction variable
      // outside the parallel loop, we use its original value (ValueToStringExpr)
      string sub_expr = EmitRefNodeAccessExprAtIdx(Index, isInParallel);
      if (sub_expr.empty()) {
        params.clear();
        break;
      }
      ss << sub_expr;
      if (i < RefNode->getSubscripts().size()-1) {
        ss << ",";
      }
      i++;
    }
    ss << "}";
    params.push_back(ss.str());
  }
  return params;
}


string ModelCodeGenProWrapperPass::EmitBranchCondExpr(BranchTNode *Branch)
{
  TranslateStatus status = SUCCESS;
  string cond = "true";
  stringstream  ss;
//  Translator->ValueToStringExpr(Branch->getCondition(), status);
  CmpInst *CI = dyn_cast<CmpInst>(Branch->getCondition());
  string tmp = "";
  if (Branch->neighbors[0]->neighbors.empty()) {
    tmp = (ReplaceInductionVarExprByType(CI->getOperand(0), OMP_PARALLEL, status)
           + Translator->PredicateToStringExpr(CI->getInversePredicate(), status)
           + ReplaceInductionVarExprByType(CI->getOperand(1), OMP_PARALLEL, status));
  } else {
    tmp = (ReplaceInductionVarExprByType(CI->getOperand(0), OMP_PARALLEL, status)
           + Translator->PredicateToStringExpr(CI->getPredicate(), status)
           + ReplaceInductionVarExprByType(CI->getOperand(1), OMP_PARALLEL, status));
  }
  if (status == SUCCESS) {
    cond = tmp;
  }
  ss << "if (" << cond << ")";
  return ss.str();
}


string ModelCodeGenProWrapperPass::EmitLoopNodeExpr(LoopTNode *LoopNode,
                                                    bool isSampledLoop)
{
  TranslateStatus status = SUCCESS;
  LoopBound *LB = LoopNode->getLoopBound();
  string induction_expr = Translator->ValueToStringExpr(LoopNode->getInductionPhi(),
                                                        status);
  string loop_init_expr = induction_expr + "_lb",
      induction_comp_predicate = "<",
      loop_final_expr = induction_expr + "_ub",
      loop_step = induction_expr + "++";
  stringstream ss;
  if (LB) {
    // the induction variable may not always the LHS of the condition
    // i.e. for (c1 = 0; c0 > c1; c1++)
    // we need to let the loop bound tell the codegen
    // where we should put the induction variable
    if (SamplingMethod == RANDOM_START && isSampledLoop) {
      loop_init_expr = induction_expr + "_start";
    } else {
      Translator->ValueToStringExpr(LB->InitValue, status);
      if (status == SUCCESS) {
        loop_init_expr = Translator->ValueToStringExpr(LB->InitValue, status);
      }
    }
    Translator->PredicateToStringExpr(LB->Predicate, status);
    if (status == SUCCESS)
      induction_comp_predicate = Translator->PredicateToStringExpr(LB->Predicate, status);
    if (!LB->FinalValue) {
      Translator->SCEVToStringExpr(LB->FinalValueSCEV, LoopNode->getLoop(), status);
      if (status == SUCCESS)
        loop_final_expr = Translator->SCEVToStringExpr(LB->FinalValueSCEV, LoopNode->getLoop(),
                                                       status);
    } else {
      Translator->ValueToStringExpr(LB->FinalValue, status);
      if (status == SUCCESS)
        loop_final_expr = Translator->ValueToStringExpr(LB->FinalValue, status);
    }
    Translator->ValueToStringExpr(LB->StepInst, status);
    if (status == SUCCESS)
      loop_step = Translator->ValueToStringExpr(LB->StepInst, status);
  }  else {
    LLVM_DEBUG(dbgs() << "No parsable Loop Bound\n");
    status = NOT_TRANSLATEABLE;
  }
  ss << "for (int " << induction_expr << "=" << loop_init_expr << ";";
//  if () {
//    ss << "!(";
//  }
  if (LB->isLHS()) {
    ss << induction_expr << induction_comp_predicate << loop_final_expr << ";";
  } else {
    ss << loop_final_expr << induction_comp_predicate << induction_expr << ";";
  }
//  if (/* */) {
//    ss << ")";
//  }
  ss << induction_expr << "=" << loop_step << ")";
  return ss.str();
}

string ModelCodeGenProWrapperPass::EmitOneParallelIterationTripFormula(LoopTNode *LoopNode)
{
  if (!LoopNode)
    return "0";
  stringstream ss;
  // get the trip count expression of one parallel loop iteration
  unsigned reference_cnt = 0;
  ss << "(";
  for (auto neighbor: LoopNode->neighbors) {
    if (RefTNode *Ref = dynamic_cast<RefTNode *>(neighbor))
      reference_cnt+=1;
    if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(neighbor))
      ss << EmitTripFormula(Loop) << "+";
    if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(neighbor)) {
      if (!Branch->neighbors[0]->neighbors.empty())
        ss << EmitTripFormula(Branch->neighbors[0]) << "+";
      if (!Branch->neighbors[1]->neighbors.empty())
        ss << EmitTripFormula(Branch->neighbors[1]) << "+";
    }
  }
  ss << reference_cnt << ")";
#if 0
  // replace the name of the parallel loop induction variable to the
  // progres[tid][i]
  string ret = ss.str();
  int distance = isInsideParallelLoopNest(LoopNode->getInductionPhi());
  string to_replace = ("progress[tid_to_run]->iteration[" + to_string(distance) + "]");
  if (distance >= 0) {
    size_t start_pos = 0;
    while((start_pos = ret.find(InductionVarNameTable[LoopNode->getInductionPhi()], start_pos)) != std::string::npos) {
      ret.replace(start_pos, InductionVarNameTable[LoopNode->getInductionPhi()].length(), to_replace);
      start_pos += to_replace.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
  }
#endif
  return ss.str();
}

string ModelCodeGenProWrapperPass::EmitTripFormula(SPSTNode *Node)
{
  string trip_formula = "";
  if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(Node)) { // compute the trip of a loop
    // assume the given loop node has no induction variable
    // dependencies and no branches inside.
    // induction variable dependencies means the loop bound of an inner-loop
    // depends on the induction variable of its parent
    //
    // (num-accesses + (subloop access)) * trip_count
    unsigned ref_count = 0;
    trip_formula += "(";
    for (auto neighbor : Loop->neighbors) {
      if (LoopTNode *SubLoopNode = dynamic_cast<LoopTNode *>(neighbor)) {
        trip_formula += (EmitTripFormula(SubLoopNode) + "+");
      } else if (dynamic_cast<RefTNode *>(neighbor)) {
        ref_count++;
      }
    }
    trip_formula += to_string(ref_count) + ")";

    TranslateStatus status = SUCCESS;
    LoopBound *LB = Loop->getLoopBound();
    string induction_expr = InductionVarNameTable[Loop->getInductionPhi()];
    string induction_init = induction_expr + "_lb",
        induction_final = induction_expr + "_ub",
        induction_step = induction_expr + "1";
    stringstream ss;
    if (LB) {
      // the induction variable may not always the LHS of the condition
      // i.e. for (c1 = 0; c0 > c1; c1++)
      // we need to let the loop bound tell the codegen
      // where we should put the induction variable
      Translator->ValueToStringExpr(LB->InitValue, status);
      if (status == SUCCESS)
        induction_init = Translator->ValueToStringExpr(LB->InitValue, status);
      if (LB->FinalValue) {
        Translator->ValueToStringExpr(LB->FinalValue, status);
        if (status == SUCCESS)
          induction_final = Translator->ValueToStringExpr(LB->FinalValue, status);
      }
      Translator->ValueToStringExpr(LB->StepValue, status);
      if (status == SUCCESS)
        induction_step = Translator->ValueToStringExpr(LB->StepValue, status);

      switch(LB->Predicate) {
      case CmpInst::ICMP_SGE:
      case CmpInst::ICMP_SLE:
      case CmpInst::ICMP_UGE:
      case CmpInst::ICMP_ULE:
        induction_final += "+1";
        break;
      default:
        break;
      }
    }
    ss << "((" << induction_final << "-" << induction_init << ")/"
       << induction_step << ")";
    trip_formula = trip_formula + "*" + ss.str();
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Node)) { // compute the trip of one branch
    unsigned ref_count = 0;
    trip_formula += "(";
    for (auto neighbor : Dummy->neighbors) {
      if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(neighbor)) {
        trip_formula += (EmitTripFormula(LoopNode) + "+");
      } else if (dynamic_cast<RefTNode *>(neighbor)) {
        ref_count++;
      }
    }
    trip_formula += to_string(ref_count) + ")";
  }
  LLVM_DEBUG(dbgs() << "trip formula is " << trip_formula << "\n");
  return trip_formula;
}

string ModelCodeGenProWrapperPass::EmitLoopTripExpr(LoopTNode *L)
{
  string loop_induction = "", loop_init_expr = "0", loop_final_expr = "N", loop_step = "1";
  string tmp = "";
  TranslateStatus status = SUCCESS;
  tmp = Translator->ValueToStringExpr(L->getLoopBound()->InitValue, status);
  if (status == SUCCESS)
    loop_init_expr = tmp;
  tmp = Translator->ValueToStringExpr(L->getLoopBound()->FinalValue, status);
  if (status == SUCCESS)
    loop_final_expr = tmp;
  tmp = Translator->ValueToStringExpr(L->getLoopBound()->StepValue, status);
  if (status == SUCCESS)
    loop_step = tmp;
  stringstream ss;
  switch(L->getLoopBound()->Predicate) {
  case llvm::CmpInst::ICMP_UGE:
  case llvm::CmpInst::ICMP_ULE:
  case llvm::CmpInst::ICMP_SGE:
  case llvm::CmpInst::ICMP_SLE:
    ss << "((" << loop_final_expr << "-" << loop_init_expr << "+1)/" << loop_step << ")";
    break;
  default:
    ss << "((" << loop_final_expr << "-" << loop_init_expr << ")/" << loop_step << ")";
    break;
  }
  return ss.str();
}


// check if the given node can be represented by the trip formula
bool ModelCodeGenProWrapperPass::canBeSimplifiedByFormula(SPSTNode *Node)
{
  if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(Node)) {
    if (QueryEngine->hasBranchInside(LoopNode))
      return false; // cannot simplify if there is branch
    // check if any of these subloops has the induction variable dependences of
    // the node;
    SmallVector<LoopTNode *, 64> SubLoops;
    QueryEngine->GetSubLoopTNode(LoopNode, SubLoops);
    for (auto ivdep : getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
        .InductionVarDepForest) {
      if (ivdep->isNodeOf(LoopNode->getInductionPhi()))
        return false;
    }
    // no subloop depends on Node, but it is possible that their subloop has
    // induction variable dependences
    for (auto SubL : SubLoops) {
      if (!canBeSimplifiedByFormula(SubL))
        return false;
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Node)) {
    for (auto neighbor : Dummy->neighbors) {
      if (!canBeSimplifiedByFormula(neighbor))
        return false;
    }
  } else if (dynamic_cast<DummyTNode *>(Node)) {
    return true;
  } else if (dynamic_cast<BranchTNode *>(Node)) {
    return false;
  }
  return true;
}

void ModelCodeGenProWrapperPass::ModelUtilGen()
{
  CodeGen->EmitCode("void simulate_negative_binomial(int thread_cnt, long n, unordered_map<long, double> &dist)");
  CodeGen->EmitCode("{");
  CodeGen->tab_count++;
  CodeGen->EmitDebugInfo("cout << \"Distribute thread-local Reuse: \" << n << endl");
  CodeGen->EmitStmt("double p = 1.0 / thread_cnt");
  CodeGen->EmitCode("if (n >= (4000. * (thread_cnt-1)) / thread_cnt) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("int i = (int)log2(n)");
  CodeGen->EmitStmt("long bin = (long)(pow(2.0, i))");
  CodeGen->EmitStmt("dist[THREAD_NUM*bin] = 1.0");
  CodeGen->EmitStmt("return");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->EmitStmt("uint64_t k = 0");
  CodeGen->EmitStmt("double nbd_prob = 0.0, prob_sum = 0.0");
  CodeGen->EmitCode("while (true) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("nbd_prob = gsl_ran_negative_binomial_pdf(k, p, n)");
  CodeGen->EmitStmt("prob_sum += nbd_prob");
  CodeGen->EmitStmt("dist[k+n] = nbd_prob");
  CodeGen->EmitCode("if (prob_sum > 0.999)");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("break");
  CodeGen->tab_count--;
  CodeGen->EmitStmt("k += 1");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of void simulate_negative_binomial()");
#if 0
  CodeGen->EmitCode("void negative_binomial_approximation(double p, uint64_t n, unordered_map<uint64_t, double> &dist)");
  CodeGen->EmitCode("{");
  CodeGen->tab_count++;
  /*
  int i = (int)log2(n);
  uint64_t bin = (uint64_t)(pow(2.0, i)*2-1), prev_bin = 0;
  double prob_sum = 0.0;
  while (true) {
    bin = (uint64_t)(pow(2.0, i)*2-1);
    if (first bin) {
      p = gsl_cdf_negative_binomial_P(bin, p, n);
    } else {
      p = gsl_cdf_negative_binomial_P(bin, p, n) - gsl_cdf_negative_binomial_P(prev_bin, p, n);
    }
    dist[bin] = p;
    prob_sum += p;
    if (prob_sum > 0.999)
      break;
    prev_bin = bin;
    i+=1;
  }
  */
  CodeGen->EmitStmt("int i = (int)log2(n)");
  CodeGen->EmitStmt("double cdf = 0.0, prob_sum = 0.0");
  CodeGen->EmitStmt("bool first_bin = true");
  CodeGen->EmitStmt("uint64_t bin = 0, prev_bin = 0");
  CodeGen->EmitComment("set the epsilon = 0.001");
  CodeGen->EmitCode("if (n >= (4000. * (THREAD_NUM-1)) / THREAD_NUM) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("bin = (uint64_t)(pow(2.0, i))");
  CodeGen->EmitStmt("dist[n*THREAD_NUM] = 1.0");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->EmitCode("while (true) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("bin = (uint64_t)(pow(2.0, i)*2-1)");
  CodeGen->EmitCode("if (first_bin) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("cdf = gsl_cdf_negative_binomial_P(bin-n, p, n)");
  CodeGen->EmitDebugInfo("cout << \"bin[-oo, \" << bin << \"] = \" << cdf << endl");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} else {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("cdf = gsl_cdf_negative_binomial_P(bin-n, p, n) - gsl_cdf_negative_binomial_P(prev_bin-n, p, n)");
  CodeGen->EmitDebugInfo("cout << \"bin[\" << prev_bin+1 << \", \" << bin << \"] = \" << cdf << endl");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->EmitStmt("dist[bin] = cdf");
  CodeGen->EmitStmt("prob_sum += cdf");
  CodeGen->EmitCode("if (prob_sum > 0.999)");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("break");
  CodeGen->tab_count--;
  CodeGen->EmitStmt("prev_bin = bin");
  CodeGen->EmitStmt("i+=1");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of void negative_binomial_approximation()");
#endif
  if (EnableParallelOpt) {
    CodeGen->EmitCode(
        "void no_share_distribute(unordered_map<long, double>& histogram_to_distribute, unordered_map<long, double>& target_histogram, int thread_cnt=THREAD_NUM)");
  } else {
    CodeGen->EmitCode(
        "void no_share_distribute(unordered_map<long, double>& histogram_to_distribute, int thread_cnt=THREAD_NUM)");
  }
  CodeGen->EmitCode("{");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("unordered_map<long, double> dist");
  CodeGen->EmitCode("for (auto entry : histogram_to_distribute) {");
  CodeGen->tab_count++;
  CodeGen->EmitCode("if (entry.first < 0) {");
  CodeGen->tab_count++;
  if (EnableParallelOpt) {
    CodeGen->EmitFunctionCall("pluss_parallel_histogram_update",
                              {"target_histogram", "entry.first", "entry.second"});
  } else {
    CodeGen->EmitFunctionCall("pluss_histogram_update", {"entry.first", "entry.second"});
  }
  CodeGen->EmitStmt("continue");
  CodeGen->tab_count--;
  CodeGen->EmitStmt("}");
  CodeGen->EmitCode("if (thread_cnt > 1) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("simulate_negative_binomial(thread_cnt, entry.first, dist)");
  CodeGen->EmitCode("for (auto dist_entry : dist) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("long ri_to_distribute = dist_entry.first");
  if (EnableParallelOpt) {
    CodeGen->EmitFunctionCall("pluss_parallel_histogram_update",
                              {"target_histogram", "ri_to_distribute", "entry.second * dist_entry.second"});
  } else {
    CodeGen->EmitFunctionCall("pluss_histogram_update",
                              {"ri_to_distribute", "entry.second * dist_entry.second"});
  }
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->EmitStmt("dist.clear()");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} else {");
  CodeGen->tab_count++;
  if (EnableParallelOpt) {
    CodeGen->EmitFunctionCall(
        "pluss_parallel_histogram_update",
        {"target_histogram", "entry.first", "entry.second"});
  } else{
    CodeGen->EmitFunctionCall(
        "pluss_histogram_update",
        {"entry.first", "entry.second"});
  }
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of if(thread_cnt > 1)");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of void no_share_distribute()");
  if (EnableParallelOpt) {
    CodeGen->EmitCode("void share_distribute(unordered_map<int, unordered_map<long, double>>&histogram_to_distribute, unordered_map<long, double> &target_histogram, int thread_cnt=THREAD_NUM)");
  } else {
    CodeGen->EmitCode("void share_distribute(unordered_map<int, unordered_map<long, double>>&histogram_to_distribute, int thread_cnt=THREAD_NUM)");
  }
  CodeGen->EmitCode("{");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("unordered_map<int, double> prob");
  CodeGen->EmitStmt("unordered_map<long, double> dist");
  CodeGen->EmitCode("for (auto share_entry : histogram_to_distribute) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("int i = 1");
  CodeGen->EmitStmt("double prob_sum = 0.0, n = (double)share_entry.first");
  CodeGen->EmitCode("for (auto reuse_entry : share_entry.second) {");
  CodeGen->tab_count++;
  CodeGen->EmitCode("if (thread_cnt > 1) {");
  CodeGen->tab_count++;
#if 0
  CodeGen->EmitCode("if (entry.first > 500) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("negative_binomial_approximation(1.0/THREAD_NUM, entry.first, dist)");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} else {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("simulate_negative_binomial(1.0/THREAD_NUM, entry.first, dist)");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
#endif
  CodeGen->EmitStmt("simulate_negative_binomial(1.0/THREAD_NUM, reuse_entry.first, dist)");
  CodeGen->EmitCode("for (auto dist_entry: dist) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("long ri_to_distribute = dist_entry.first");
#if 0
  CodeGen->EmitCode("if (entry.first <= 500) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("ri_to_distribute = entry.first + dist_entry.first");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
#endif
  CodeGen->EmitStmt("double cnt_to_distribute = reuse_entry.second * dist_entry.second");
  CodeGen->EmitDebugInfo("printf(\"new ri: \t%lu, %lu*%f=%f\\n\", ri_to_distribute, reuse_entry.second, dist_entry.second, cnt_to_distribute)");
  CodeGen->EmitStmt("prob_sum = 0.0");
  CodeGen->EmitStmt("i = 1");
  CodeGen->EmitCode("while (true) {");
  CodeGen->tab_count++;
  CodeGen->EmitCode("if (pow(2.0, (double)i) > ri_to_distribute) { break; }");
  CodeGen->EmitStmt("prob[i] = pow(1-(pow(2.0, (double)i-1) / ri_to_distribute), n-1) - pow(1-(pow(2.0, (double)i) / ri_to_distribute), n-1)");
  CodeGen->EmitStmt("prob_sum += prob[i]");
  CodeGen->EmitDebugInfo("printf(\"prob[2^%d <= ri < 2^%d] = %f (%f)\\n\", i-1, i, prob[i], prob[i]*cnt_to_distribute)");
  CodeGen->EmitStmt("i++");
//    CodeGen->EmitStmt("cout << prob_sum << endl");
  CodeGen->EmitCode("if (prob_sum == 1.0) { break; }");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of while(true)");
  CodeGen->EmitCode("if (prob_sum != 1.0) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("prob[i-1] = 1 - prob_sum");
  CodeGen->EmitDebugInfo("printf(\"prob[ri >= 2^%d] = %f (%f)\\n\", i-1, prob[i], prob[i]*cnt_to_distribute)");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->EmitCode("for (auto bin : prob) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("long new_ri = (long)pow(2.0, bin.first-1)");
  if (EnableParallelOpt) {
    CodeGen->EmitStmt("pluss_parallel_histogram_update(target_histogram,new_ri,bin.second*cnt_to_distribute)");
  } else {
    CodeGen->EmitStmt("pluss_histogram_update(new_ri,bin.second*cnt_to_distribute)");
  }
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->EmitStmt("prob.clear()");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of iterating dist");
  CodeGen->EmitStmt("dist.clear()");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} else {");
  CodeGen->tab_count++;
  if (EnableParallelOpt) {
    CodeGen->EmitFunctionCall(
        "pluss_parallel_histogram_update",
        {"target_histogram", "reuse_entry.first", "reuse_entry.second"});
  } else {
    CodeGen->EmitFunctionCall(
        "pluss_histogram_update",
        {"reuse_entry.first", "reuse_entry.second"});
  }
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of iterating all reuse entries");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of iterating all type of reuses");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of void share_distribute()");
}

void ModelCodeGenProWrapperPass::HeaderGen()
{
  CodeGen->EmitCode("#include <iostream>");
  CodeGen->EmitCode("#include <cmath>");
  CodeGen->EmitCode("#include <random>");
  CodeGen->EmitCode("#include <algorithm>");
  CodeGen->EmitCode("#include <queue>");
  if (EnableParallelOpt) {
    CodeGen->EmitCode("#include <thread>");
    CodeGen->EmitCode("#include <sched.h>");
    CodeGen->EmitCode("#include <pthread.h>");
  }
  if (InterleavingTechnique == 1) {
    CodeGen->EmitCode("#include <numeric>");
    CodeGen->EmitCode("#include <gsl/gsl_rng.h>");
    CodeGen->EmitCode("#include <gsl/gsl_randist.h>");
    CodeGen->EmitCode("#include <gsl/gsl_cdf.h> // cdf of gaussian distribution");
    CodeGen->EmitCode("#include <time.h>");
  }
  CodeGen->EmitCode("#include \"pluss.h\"");
  CodeGen->EmitCode("#include \"pluss_utils.h\"");
  CodeGen->EmitStmt("using namespace std");
  if ((EnableSampling && SamplingMethod == RANDOM_START) || EnablePerReference) {
    CodeGen->EmitStmt(
        "unordered_map<string, long> iteration_traversed_map");
    if (!EnableParallelOpt) {
      CodeGen->EmitStmt(
          "unordered_map<long, double> no_share_intra_thread_RI");
      CodeGen->EmitStmt(
          "unordered_map<int, unordered_map<long, double>> share_intra_thread_RI");
    }
  } else {
    CodeGen->EmitStmt("long max_iteration_count = 0");
  }
}

void ModelCodeGenProWrapperPass::RefAddressFuncGen()
{
  unordered_map<string, vector<string>> ArrayBound
      = getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>().ArrayBound;
  stringstream ss;
  for (auto entry : SPSNodeNameTable) {
    if (RefTNode *RefNode = dynamic_cast<RefTNode *>(entry.first)) {
      ss << "uint64_t GetAddress_" << entry.second << "(";
      unsigned i = 0;
      for (; i < RefNode->getSubscripts().size(); i++) {
        ss << "int idx" << to_string(i);
        if (i < RefNode->getSubscripts().size()-1)
          ss << ",";
      }
      ss << ") {";
      CodeGen->EmitCode(ss.str());
      ss.str("");
      CodeGen->tab_count++;
      if (RefNode->getSubscripts().empty()) {
        CodeGen->EmitStmt("return 0");
      } else {
        i = 0;
        ss << "uint64_t addr_" << SPSNodeNameTable[RefNode] << " = ";
        for (; i < RefNode->getSubscripts().size(); i++) {
          string bound = "1";
          unsigned j = i + 1;
          for (; j < RefNode->getSubscripts().size(); j++) {
            if (j == i + 1)
              bound = ArrayBound[RefNode->getArrayNameString()][j];
            else
              bound += ArrayBound[RefNode->getArrayNameString()][j];
            if (j > 0 && j < RefNode->getSubscripts().size() - 1)
              bound += "*";
          }
          if (i < RefNode->getSubscripts().size() - 1)
            ss << "(idx" << to_string(i) << "*" << bound << ")+";
          else
            ss << "(idx" << to_string(i) << "*" << bound << ")";
        }
        CodeGen->EmitStmt(ss.str());
        ss.str("");
        CodeGen->EmitStmt("return addr_" + SPSNodeNameTable[RefNode] +
                          "*DS/CLS");
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
    }
  }
}

void ModelCodeGenProWrapperPass::MainFuncGen()
{
  CodeGen->EmitCode("int main() {");
  CodeGen->tab_count++;
  // add timer
#if 0
  CodeGen->EmitFunctionCall("pluss_timer_init", {});
  CodeGen->EmitComment("Init the locality analysis library");
  CodeGen->EmitFunctionCall("pluss_init", {});
#endif
  // random_start sampling must do per-ref analysis
  if ((EnableSampling && (SamplingMethod == RANDOM_START)) || EnablePerReference) {
    CodeGen->EmitStmt("long max_iteration_count = 0");
    if (EnableParallelOpt) {
      for (auto node : SPSNodeNameTable) {
        if (RefTNode *ref = dynamic_cast<RefTNode *>(node.first)) {
          CodeGen->EmitStmt("unordered_map<long, double> histogram_" + node.second);
        }
      }
    }
    CodeGen->EmitFunctionCall("pluss_timer_start", {});
    unsigned coreid = 0, loopid = 0;
    vector<string> references_to_traverse;
    for (auto Loop : LoopNodes) {
      if (Loop->isParallelLoop()) {
        for (auto node : SPSNodeNameTable) {
          if (QueryEngine->GetParallelLoopDominator(node.first) == Loop) {
            if (RefTNode *ref = dynamic_cast<RefTNode *>(node.first)) {
              // top level reference will not be analyzed, we should skip those sampler function call top level reference does not have dominator
              if (!QueryEngine->GetImmdiateLoopDominator(ref)) {
                CodeGen->EmitComment("Skip sampler " + ref->getRefExprString());
                continue;
              }
              if (EnableParallelOpt) {
                CodeGen->EmitFunctionCall("thread t_sampler_"+node.second,
                                          {"sampler_"+node.second, "ref(histogram_"+node.second+")"});
                CodeGen->EmitComment("set the thread affinity");
                CodeGen->EmitStmt("cpu_set_t cpuset_"+node.second);
                CodeGen->EmitStmt("CPU_ZERO(&cpuset_"+node.second+")");
                if ((coreid % 40) < 20)
                  CodeGen->EmitFunctionCall("CPU_SET", {to_string(coreid*2),
                                                        "&cpuset_"+node.second});
                else if ((coreid % 40) >= 20)
                  CodeGen->EmitFunctionCall("CPU_SET", {to_string((coreid%20)*2+1),
                                                        "&cpuset_"+node.second});
                CodeGen->EmitFunctionCall("pthread_setaffinity_np", {"t_sampler_"+node.second+".native_handle()",
                                                                     "sizeof(cpu_set_t)",
                                                                     "&cpuset_"+node.second});
              } else {
                CodeGen->EmitFunctionCall("sampler_" + node.second, {});
              }
              references_to_traverse.emplace_back(node.second);
              coreid++;
            }
          }
        }
        loopid += 1;
      }
    }
    if (EnableParallelOpt) {
      // generate the join function
      for (auto refname : references_to_traverse) {
        CodeGen->EmitStmt("t_sampler_"+refname+".join()");
      }
      CodeGen->EmitComment("Merge the histogram found by each reference");
      for (auto refname : references_to_traverse) {
        CodeGen->EmitCode("for (auto pair : histogram_"+refname+") {");
        CodeGen->tab_count++;
        CodeGen->EmitFunctionCall("pluss_histogram_update",{"pair.first","pair.second"});
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
      }
    } else {
      CodeGen->EmitFunctionCall("no_share_distribute",{"no_share_intra_thread_RI"});
      CodeGen->EmitFunctionCall("share_distribute",{"share_intra_thread_RI"});
    }
  } else {
    CodeGen->EmitFunctionCall("pluss_timer_start", {});
    CodeGen->EmitFunctionCall("sampler", {});
  }
  CodeGen->EmitFunctionCall("pluss_AET", {});
  CodeGen->EmitFunctionCall("pluss_timer_stop", {});
  CodeGen->EmitFunctionCall("pluss_timer_print", {});
  CodeGen->EmitFunctionCall("pluss_print_histogram", {});
  CodeGen->EmitFunctionCall("pluss_print_mrc", {});
  if ((EnableSampling && (SamplingMethod == RANDOM_START)) || EnablePerReference) {
    CodeGen->EmitCode("for (auto entry: iteration_traversed_map) {");
    CodeGen->tab_count++;
//    CodeGen->EmitStmt("cout << entry.first << \" touches \" << entry.second << \" iterations\" << endl");
    CodeGen->EmitStmt(
        "max_iteration_count = max(max_iteration_count, entry.second)");
    CodeGen->tab_count--;
    CodeGen->EmitCode("}");
  }
  CodeGen->EmitStmt("cout << \"max iteration traversed\" << endl");
  CodeGen->EmitStmt("cout << max_iteration_count << endl");
  CodeGen->EmitStmt("return 0");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  return;
}

void ModelCodeGenProWrapperPass::SamplerBodyGen(bool isPerReference)
{
//  CodeGen->EmitChunkDispatcher(STATIC);
//  CodeGen->EmitProgressClassDef();
  if (EnableSampling) {
    CodeGen->EmitComment("Generate sampler with sampling");
    stringstream sampler_typess;
    switch(SamplingMethod) {
    case RANDOM_START:
      sampler_typess << "Use random start sampling with ratio "
                     << SamplingRatio << "%";
      break;
    case SEQUENTIAL_START:
      sampler_typess << "Use serial start sampling with ratio "
                     << SamplingRatio << "%";
      break;
    default:
      sampler_typess << "WARNING, unrecognized sampling approach!";
      break;
    }
    CodeGen->EmitComment(sampler_typess.str());
  } else {
    CodeGen->EmitComment("Generate sampler without sampling");
  }

  for (auto ref : SPSNodeNameTable) {
    if (RefTNode *TargetRef = dynamic_cast<RefTNode *>(ref.first)) {
#if 0
      // if using sampling, we will generate code to
        // 1) construct all samplings
        // 2) sort these samples in order (smaller iteration serves first)
        // 3) build a while loop outside that examines whether the vector stores
        // all samples are empty.
        //
        // when we find a reuse of one of the samples during the traversal,
        // we remove it from the list
        if (SamplingRatio < 100) {
          CodeGen->EmitComment("Generate sampling for " +
                               SPSNodeNameTable[TargetRef]);
          CodeGen->EmitCode("while (!.empty) {");
          CodeGen->tab_count++;
        }
#endif
      // we get all LoopNode that can be backtraced from the TargetRef
      // if TargetRef is the top access (not in a loop), its path should be
      // empty
      AccPath path;
      QueryEngine->GetPath(TargetRef, NULL, path);
      if (path.empty()) {
        CodeGen->EmitComment("Skip generaring the sampler for "
                             "reference " + TargetRef->getRefExprString());
        LLVM_DEBUG(dbgs() << "SKIP sampler for reference "
                          << SPSNodeNameTable[TargetRef] << "\n");
        continue;
      }
#if 0
      auto path_riter = path.rbegin();
        while (path_riter != path.rend()) {
          if (path_riter == path.rbegin()) {
            // outmost loop
          }
          path_riter++;
        }
#endif
      CodeGen->EmitComment(TargetRef->getRefExprString());
      if (EnableParallelOpt)
        CodeGen->EmitCode("void sampler_" + ref.second + "(unordered_map<long, double> &histogram) {");
      else
        CodeGen->EmitCode("void sampler_" + ref.second + "() {");
      CodeGen->tab_count++;
      stringstream ss;
      auto topiter = getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
          .TreeRoot->neighbors.begin();
      CodeGen->EmitComment("Declare components will be used in Parallel RI search");
      CodeGen->EmitStmt("array<Progress *, THREAD_NUM> progress = { nullptr }");
      CodeGen->EmitStmt("vector<int> idle_threads, worker_threads, subscripts");
      CodeGen->EmitStmt("ChunkDispatcher dispatcher");
      CodeGen->EmitStmt("int tid_to_run = 0, start_tid = 0, working_threads = THREAD_NUM");
      CodeGen->EmitStmt("uint64_t addr = 0");
      CodeGen->EmitComment("global counter");
      CodeGen->EmitStmt("array<long, THREAD_NUM> count = {0}");
      CodeGen->EmitStmt("unordered_map<int, unordered_map<uint64_t, long>> LAT");
      CodeGen->EmitStmt("unordered_map<int, unordered_map<long, Iteration *>> LATSampleIterMap");
      if (EnableParallelOpt) {
        CodeGen->EmitStmt(
            "unordered_map<long, double> no_share_histogram");
        CodeGen->EmitStmt("unordered_map<int, unordered_map<long, double>> share_histogram");
      }
//        LLVM_DEBUG(
//            for (auto l: path) {
//              if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(l)) {
//                dbgs() << Loop->getLoopStringExpr() << "\n";
//              }
//            }
//            );
      for (; topiter != getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
          .TreeRoot->neighbors.end(); topiter++) {
        assert(!path.empty() && "Reference that has the sampler func"
                                " must be inside a loop");
        // TargetRef is inside a loop nest, we find which loop nest has
        // TargetRef
        if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(*topiter)) {
//            LLVM_DEBUG(dbgs() << "check loop" << Loop->getLoopStringExpr() << "\n");
          if (Loop != *(path.rbegin())) {
            CodeGen->EmitComment("Skip loop " + Loop->getLoopStringExpr());
            continue;
          } else if (EnableSampling && SamplingMethod == RANDOM_START) {
            if (Loop == *(path.rbegin())) {
              // we need to generate all samplings here
              CodeGen->EmitStmt("unsigned s=0, traversing_samples=0");
              CodeGen->EmitStmt("bool start_reuse_search=false");
              CodeGen->EmitStmt("Iteration start_iteration");
              CodeGen->EmitStmt("priority_queue<Iteration, vector<Iteration>, IterationComp> samples");
              CodeGen->EmitStmt("unordered_set<string> sample_names, samples_meet");
#if 0
              CodeGen->EmitStmt("mt19937 rng(dev())");
#endif
              if (EnableParallelOpt) {
                CodeGen->EmitStmt("hash<std::thread::id> hasher");
                CodeGen->EmitStmt(
                    "static thread_local mt19937 generator(time(NULL)+hasher(this_thread::get_id()))");
              } else {
                CodeGen->EmitStmt("srand(time(NULL))");
              }
              CodeGen->EmitComment("generate all samples and sort "
                                   "them in order");
              if (PerReferenceSampleCnt.find(TargetRef) != PerReferenceSampleCnt.end()) {
                CodeGen->EmitStmt("int num_samples = " +
                                  to_string(PerReferenceSampleCnt[TargetRef]));

              } else {
                CodeGen->EmitStmt("int num_samples = NUM_SAMPLES_"+SPSNodeNameTable[TargetRef]);
              }
#if 0
              // If a loop bound is constant, we can declare the random number generators
                // before entering the loop and we do not have to check if the generated sample
                // is okay (exceed the bound)
                SmallVector<LoopTNode *, 4> ConstantBoundLoopsInPath;
                auto path_riter = path.rbegin();
                while (path_riter != path.rend()) {
                  LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
                  assert(LoopInPath && "AccPath element should be LoopTNode type");
                  if (QueryEngine->hasConstantLoopBound(LoopInPath)) {
                    ConstantBoundLoopsInPath.push_back(LoopInPath);
                  }
                  path_riter++;
                }
                if (!ConstantBoundLoopsInPath.empty()) {
                  // generate the random number generator before the while loop
                  // if the bound does not have any dependenies
                  path_riter = path.rbegin();
                  while (path_riter != path.rend()) {
                    LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
                    assert(LoopInPath && "AccPath element should be LoopTNode type");
                    if (find(ConstantBoundLoopsInPath.begin(), ConstantBoundLoopsInPath.end(),
                             LoopInPath) == ConstantBoundLoopsInPath.end()) {
                      path_riter++;
                      continue;
                    }
                    TranslateStatus status = SUCCESS;
                    string induction_expr =
                        InductionVarNameTable[LoopInPath->getInductionPhi()],
                        loop_init_expr = induction_expr + "_lb",
                        loop_final_expr = induction_expr + "_ub", loop_step = "1";
                    // need to handle the case where the loop iteration range
                    // depends on one of its parent loop
                    pair<LoopTNode *, LoopTNode *> IndvarDepResult =
                        CheckDependenciesInPath(LoopInPath, path);
                    string tmp = Translator->ValueToStringExpr(
                        LoopInPath->getLoopBound()->InitValue, status);
                    if (status == SUCCESS) {
                      loop_init_expr = tmp;
                    }
                    if (IndvarDepResult.first) { // has lower dependency
                      ReplaceSubstringWith(
                          loop_init_expr,
                          InductionVarNameTable[IndvarDepResult.first->getInductionPhi()],
                          "sample_" + InductionVarNameTable[IndvarDepResult.first
                              ->getInductionPhi()]);
                    }
                    if (LoopInPath->getLoopBound()->FinalValue) {
                      string tmp = Translator->ValueToStringExpr(
                          LoopInPath->getLoopBound()->FinalValue, status);
                      if (status == SUCCESS)
                        loop_final_expr = tmp;
                      if (IndvarDepResult.second) {
                        ReplaceSubstringWith(
                            loop_final_expr,
                            InductionVarNameTable[IndvarDepResult.second->getInductionPhi()],
                            "sample_" + InductionVarNameTable[IndvarDepResult.second
                                ->getInductionPhi()]);
                      }
                    }
                    Translator->ValueToStringExpr(LoopInPath->getLoopBound()->StepValue,
                                                  status);
                    if (status == SUCCESS)
                      loop_step = Translator->ValueToStringExpr(
                          LoopInPath->getLoopBound()->StepValue, status);
                    // build the sample bound expression
                    // the bound would be
                    // [ init_val / step, final_val / step] * step + init_val
                    // or
                    // [ init_val / step, final_val / step) * step + init_val
                    //
                    // the position of init_val and final_val depends on the sign
                    // of step
                    string sample_upper_bound_expr = "";
                    sample_upper_bound_expr =
                        "((" + loop_final_expr + "-" + loop_init_expr + ")" + "/" + loop_step;
                    switch (LoopInPath->getLoopBound()->Predicate) {
                    case llvm::CmpInst::ICMP_UGT:
                    case llvm::CmpInst::ICMP_SGT:
                    case llvm::CmpInst::ICMP_ULT:
                    case llvm::CmpInst::ICMP_SLT:
                    case llvm::CmpInst::ICMP_NE:
                      sample_upper_bound_expr += "-((" + loop_final_expr + "-" +
                                                 loop_init_expr + ")%" + loop_step + "==0)";
                      break;
                    default:
                      break;
                    }
                    sample_upper_bound_expr += ")";
                    ss << "uniform_int_distribution<mt19937::result_type> dist_";
                    ss << induction_expr << "(0," << sample_upper_bound_expr << ")";
                    CodeGen->EmitStmt(ss.str());
                    ss.str(""); // clear the stream to generate other sampling stmts
                    path_riter++;
                  }
                }
#endif
              CodeGen->EmitCode("while (s < num_samples) {");
              CodeGen->tab_count++;
              CodeGen->EmitStmt("string sample_label = \""+ref.second+"\"");
              stringstream ss;
              auto path_riter = path.rbegin();
              while (path_riter != path.rend()) {
                LoopTNode *LoopInPath =
                    dynamic_cast<LoopTNode *>(*path_riter);
                assert(LoopInPath &&
                           "AccPath element should be LoopTNode type");
//                  if (LoopInPath->isParallelLoop())
//                    break;
                TranslateStatus status = SUCCESS;
                string
                    induction_expr =
                    InductionVarNameTable[LoopInPath->getInductionPhi()],
                    loop_init_expr = induction_expr + "_lb",
                    loop_final_expr = induction_expr + "_ub", loop_step = "1";

                // need to handle the case where the loop iteration range
                // depends on one of its parent loop
                pair<LoopTNode *, LoopTNode *> IndvarDepResult =
                    CheckDependenciesInPath(LoopInPath, path);
                string tmp = Translator->ValueToStringExpr(
                    LoopInPath->getLoopBound()->InitValue, status);
                if (status == SUCCESS) {
                  loop_init_expr = tmp;
                }
                if (IndvarDepResult.first) { // has lower dependency
                  ReplaceSubstringWith(loop_init_expr,
                                       InductionVarNameTable[IndvarDepResult.first->getInductionPhi()],
                                       "sample_" +
                                       InductionVarNameTable[IndvarDepResult.first->getInductionPhi()]);
                }
                if (LoopInPath->getLoopBound()->FinalValue) {
                  string tmp = Translator->ValueToStringExpr(
                      LoopInPath->getLoopBound()->FinalValue, status);
                  if (status == SUCCESS)
                    loop_final_expr = tmp;
                  if (IndvarDepResult.second) {
                    ReplaceSubstringWith(loop_final_expr,
                                         InductionVarNameTable[IndvarDepResult.second->getInductionPhi()],
                                         "sample_" +
                                         InductionVarNameTable[IndvarDepResult.second->getInductionPhi()]);
                  }
                }
                Translator->ValueToStringExpr(
                    LoopInPath->getLoopBound()->StepValue, status);
                if (status == SUCCESS)
                  loop_step = Translator->ValueToStringExpr(
                      LoopInPath->getLoopBound()->StepValue, status);
                // build the sample bound expression
                // the bound would be
                // [ init_val / step, final_val / step] * step + init_val
                // or
                // [ init_val / step, final_val / step) * step + init_val
                //
                // the position of init_val and final_val depends on the sign
                // of step
                string sample_upper_bound_expr = "";
                sample_upper_bound_expr = "((" + loop_final_expr + "-" + loop_init_expr + ")"
                                          + "/" + loop_step;
                switch (LoopInPath->getLoopBound()->Predicate) {
                case llvm::CmpInst::ICMP_UGT:
                case llvm::CmpInst::ICMP_SGT:
                case llvm::CmpInst::ICMP_ULT:
                case llvm::CmpInst::ICMP_SLT:
                case llvm::CmpInst::ICMP_NE:
                  sample_upper_bound_expr += "-((" + loop_final_expr + "-" + loop_init_expr + ")%" +
                                             loop_step + "==0)";
                  break;
                default:
                  break;
                }
                sample_upper_bound_expr += ")";
                if (path_riter != path.rbegin()) {
                  CodeGen->EmitCode("if (" + sample_upper_bound_expr +
                                    "< 0) { continue; }");
                }
#if 0
                ss << "uniform_int_distribution<mt19937::result_type> dist_";
                    ss << induction_expr << "(0," << sample_upper_bound_expr
                       << ")";
                    CodeGen->EmitStmt(ss.str());
#endif
                ss.str(""); // clear the stream to generate other sampling stmts
                //                  ss << "int sample_" << induction_expr << " = rand() % ("
                //                     << loop_final_expr << "-" << loop_init_expr << ") + "
                //                     << loop_step;
#if 0
                ss << "int sample_" << induction_expr << " = (dist_"
                     << induction_expr << "(rng)*(" << loop_step
                     << ")+(" << loop_init_expr << "))";
#endif
                ss << "int sample_" << induction_expr << " = ("
                   << sample_upper_bound_expr << " == 0)?(" << loop_init_expr << "):"
                   << "(rand()%(" << sample_upper_bound_expr << "))*(" << loop_step
                   << ")+(" << loop_init_expr << ")";
                CodeGen->EmitStmt(ss.str());
                ss.str(""); // clear the stream to generate other sampling stmts
                CodeGen->EmitDebugInfo("cout << \"sample_" + induction_expr
                                       +  ": [\" << (" + loop_init_expr
                                       + ") << \",\" << ("
                                       + sample_upper_bound_expr + "*(" + loop_step
                                       + ")+" + loop_init_expr
                                       + ") << \"] samples \" << sample_"
                                       + induction_expr + " << endl");
//                  CodeGen->EmitStmt("sample_label += (to_string(sample_" +
//                                    induction_expr + ") + \"-\")");
                path_riter++;
              }
              path_riter = path.rbegin();
              while (path_riter != path.rend()) {
                LoopTNode *LoopInPath =
                    dynamic_cast<LoopTNode *>(*path_riter);
                assert(LoopInPath &&
                           "AccPath element should be LoopTNode type");
                string
                    induction_expr =
                    InductionVarNameTable[LoopInPath->getInductionPhi()];
                CodeGen->EmitStmt("sample_label += (to_string(sample_" +
                                  induction_expr + ") + \"-\")");
                path_riter++;
              }
              CodeGen->EmitComment("avoid duplicate sampling");
              CodeGen->EmitCode(
                  "if (sample_names.find(sample_label)"
                  "!=sample_names.end()) { continue; }");
              CodeGen->EmitStmt("sample_names.insert(sample_label)");
              path_riter = path.rbegin();
              ss << "Iteration sample(\"" << ref.second + "\", {";
              // TODO: should consider the case that an reference only use one
              // loop induction variable even it inside a 2-level loop nest
              // or a reference only inside 1-level loop nest but
              // it use its induction variable as a subscript in multiple
              // array dimensions.
              // i.e.
              // for (i = 0; i < N; i++)
              //    a[i][i];
              // for (i = 0; i < N; i++)
              //   for (j = 0; j < N; j++)
              //      a[j][j]
              while (path_riter != path.rend()) {
                LoopTNode *LoopInPath =
                    dynamic_cast<LoopTNode *>(*path_riter);
                assert(LoopInPath &&
                           "AccPath element should be LoopTNode type");
                //                if (LoopInPath->isParallelLoop())
                //                  break;
                string induction_expr =
                    InductionVarNameTable[LoopInPath->getInductionPhi()];
                ss << "sample_" << induction_expr;
                if (path_riter != path.rend() - 1)
                  ss << ",";
                path_riter++;
              }
              ss << "},";
              if (LoopTNode *ParallelLoop = QueryEngine->GetParallelLoopDominator(TargetRef)) {
                pair<LoopTNode *, LoopTNode *> IndvarDepResult =
                    CheckDependenciesInPath(ParallelLoop, path);
                string induction_expr =
                    InductionVarNameTable[ParallelLoop->getInductionPhi()];
                string loop_init_expr = induction_expr + "_lb", loop_step = "1";
                TranslateStatus status = SUCCESS;
                string tmp = Translator->ValueToStringExpr(
                    ParallelLoop->getLoopBound()->InitValue, status);
                if (status == SUCCESS) {
                  loop_init_expr = tmp;
                }
                if (IndvarDepResult.first) { // has lower dependency
                  ReplaceSubstringWith(loop_init_expr,
                                       InductionVarNameTable[IndvarDepResult.first->getInductionPhi()],
                                       "sample_" +
                                       InductionVarNameTable[IndvarDepResult.first->getInductionPhi()]);
                }
                tmp = Translator->ValueToStringExpr(
                    ParallelLoop->getLoopBound()->StepValue, status);
                if (status == SUCCESS)
                  loop_step = tmp;
                AccPath pathTmp;
                QueryEngine->GetPath(ParallelLoop, nullptr, pathTmp, false);
                ss << loop_init_expr << "," << loop_step << ",true," << pathTmp.size();
              } else {
                ss << "0,1,false,0";
              }
              ss << ")";
              CodeGen->EmitStmt(ss.str());
              ss.str(""); // clear the stream to generate other sampling stmts
              CodeGen->EmitStmt("samples.push(sample)");
#if 0
              CodeGen->EmitCode("if (s == 0 || "
                                  "sample.compare(start_iteration) < 0) "
                                  "{ start_iteration = sample; }");
#endif
              CodeGen->EmitStmt("s++");
              CodeGen->tab_count--;
              CodeGen->EmitCode("} // the end of while(s < NUM_SAMPLES)");
#if 0
              CodeGen->EmitDebugInfo(
                    "cout << \"Starts from \" << start_iteration.toString() << endl");
#endif
              CodeGen->EmitCode("while (!samples.empty()) {");
              CodeGen->tab_count++;
              CodeGen->EmitLabel("START_SAMPLE_" + SPSNodeNameTable[TargetRef]);
              CodeGen->EmitStmt("start_reuse_search = false");
              CodeGen->EmitStmt("start_iteration = samples.top()");
              CodeGen->EmitStmt("traversing_samples++");
              CodeGen->EmitStmt("samples.pop()");
              CodeGen->EmitStmt("if (!idle_threads.empty()) { idle_threads.clear(); }");
              CodeGen->EmitStmt("if (!worker_threads.empty()) { worker_threads.clear(); }");
              CodeGen->EmitStmt("if (!LAT.empty()) { ");
              CodeGen->tab_count++;
              CodeGen->EmitCode("for (unsigned i = 0; i < LAT.size(); i++) {");
              CodeGen->tab_count++;
              CodeGen->EmitFunctionCall("pluss_parallel_histogram_update",
                                        {"no_share_intra_thread_RI", "-1", "LAT[i].size()"});
              CodeGen->EmitStmt("LAT.clear();");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->EmitCode("if (!LATSampleIterMap.empty()) {");
              CodeGen->tab_count++;
              CodeGen->EmitCode(
                  "for (auto LATSampleIterMapEntry : LATSampleIterMap) {");
              CodeGen->tab_count++;
              CodeGen->EmitCode(
                  "for (auto entry : LATSampleIterMapEntry.second) {");
              CodeGen->tab_count++;
              CodeGen->EmitStmt("delete entry.second");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->EmitStmt("LATSampleIterMapEntry.second.clear()");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->EmitStmt("LATSampleIterMap.clear()");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->EmitDebugInfo("cout << \"Start tracking sample \" << start_iteration.toString() << endl");
              unsigned i = 0;
              path_riter = path.rbegin();
              while (path_riter != path.rend()) {
                LoopTNode *LoopInPath =
                    dynamic_cast<LoopTNode *>(*path_riter);
                assert(LoopInPath &&
                           "AccPath element should be LoopTNode type");
                //                if (LoopInPath->isParallelLoop())
                //                  break;
                string induction_expr =
                    InductionVarNameTable[LoopInPath->getInductionPhi()];
                CodeGen->EmitStmt("int sample_" + induction_expr +
                                  "_start=start_iteration.ivs[" +
                                  to_string(i) + "]");
                CodeGen->EmitStmt("int " + induction_expr + "_start=0");
                path_riter++;
                i++;
              }
              break;
            }
          } // end of the (SamplingRatio < 100) check
          else {
            CodeGen->EmitStmt("bool start_reuse_search=false");
#if 0
            CodeGen->EmitStmt("mt19937 rng(dev())");
#endif
            if (EnableParallelOpt) {
              CodeGen->EmitStmt("hash<std::thread::id> hasher");
              CodeGen->EmitStmt(
                  "static thread_local mt19937 generator(time(NULL)+hasher(this_thread::get_id()))");
            } else {
              CodeGen->EmitStmt("srand(dev())");
            }
            break;
          }
        }
      }
      // start generate the body from the first node that could form a reuse
      // with the TargetRef
      for (; topiter != getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
          .TreeRoot->neighbors.end(); topiter++) {
        PerRefSamplerBodyGenImpl(*topiter, TargetRef);
      }

      if (EnableSampling && SamplingMethod == RANDOM_START) {
        CodeGen->EmitStmt("goto END_SAMPLE");
        CodeGen->tab_count--;
        CodeGen->EmitCode("} // end of while (!samples.empty())");
        CodeGen->EmitLabel("END_SAMPLE");
        CodeGen->EmitStmt("if (!idle_threads.empty()) { idle_threads.clear(); }");
        CodeGen->EmitStmt("if (!worker_threads.empty()) { worker_threads.clear(); }");
        CodeGen->EmitStmt("if (!LAT.empty()) { ");
        CodeGen->tab_count++;
        CodeGen->EmitCode("for (unsigned i = 0; i < LAT.size(); i++) {");
        CodeGen->tab_count++;
        CodeGen->EmitFunctionCall("pluss_parallel_histogram_update",
                                  {"no_share_intra_thread_RI", "-1", "LAT[i].size()"});
        CodeGen->EmitStmt("LAT.clear();");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->EmitCode("for (unsigned i = 0; i < progress.size(); i++) {");
        CodeGen->tab_count++;
        CodeGen->EmitCode("if (progress[i]) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("delete progress[i]");
        CodeGen->EmitStmt("progress[i] = nullptr");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->EmitCode("if (!LATSampleIterMap.empty()) {");
        CodeGen->tab_count++;
        CodeGen->EmitCode("for (auto LATSampleIterMapEntry : LATSampleIterMap) {");
        CodeGen->tab_count++;
        CodeGen->EmitCode("for (auto entry : LATSampleIterMapEntry.second) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("delete entry.second");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->EmitStmt("LATSampleIterMapEntry.second.clear()");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->EmitStmt("LATSampleIterMap.clear()");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        if (EnableParallelOpt) {
          CodeGen->EmitFunctionCall("no_share_distribute",
                                    {"no_share_histogram", "histogram"});
          CodeGen->EmitFunctionCall("share_distribute",
                                    {"share_histogram", "histogram"});
          CodeGen->EmitStmt("no_share_histogram.clear()");
          CodeGen->EmitStmt("share_histogram.clear()");
        }
        CodeGen->EmitStmt(
            "iteration_traversed_map[\"" + ref.second +
            "\"] = accumulate(count.begin(), count.end(), 0)");
//          CodeGen->EmitStmt("cout << \"RI histogram of reference \" << \"" + ref.second + "\" << endl");
//          CodeGen->EmitFunctionCall("pluss_parallel_print_histogram", {"no_share_intra_thread_RI"});
        CodeGen->EmitStmt("return");
      } else {
        CodeGen->EmitStmt("idle_threads.clear()");
        CodeGen->EmitStmt("worker_threads.clear()");
        CodeGen->EmitCode("for (unsigned i = 0; i < progress.size(); i++) {");
        CodeGen->tab_count++;
        CodeGen->EmitCode("if (progress[i]) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("delete progress[i]");
        CodeGen->EmitStmt("progress[i] = nullptr");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->EmitCode(
            "for (auto LATSampleIterMapEntry : LATSampleIterMap) {");
        CodeGen->tab_count++;
        CodeGen->EmitCode(
            "for (auto entry : LATSampleIterMapEntry.second) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("delete entry.second");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
        CodeGen->EmitStmt("LATSampleIterMapEntry.second.clear()");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");

        CodeGen->EmitStmt("LATSampleIterMap.clear()");
#if 0
        CodeGen->EmitFunctionCall("pluss_AET", {});
          CodeGen->EmitFunctionCall("pluss_print_histogram", {});
          CodeGen->EmitStmt("cout << \"Begin Loop \" << loop_cnt-1 << \"\\n\"");
          CodeGen->EmitFunctionCall("pluss_print_mrc", {});
          CodeGen->EmitStmt("cout << \"End Loop \" << loop_cnt-1 << \"\\n\"");
          CodeGen->EmitComment("reset all histograms");
          CodeGen->EmitStmt("no_share_intra_thread_RI.clear()");
          CodeGen->EmitStmt("share_intra_thread_RI.clear()");
          CodeGen->EmitFunctionCall("pluss_histogram_reset", {});
#endif
        CodeGen->EmitStmt(
            "iteration_traversed_map[\"" + ref.second +
            "\"] = accumulate(count.begin(), count.end(), 0)");
      }
      // end of the sampler function
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of the sampler function");
    }
  }
}

void ModelCodeGenProWrapperPass::SamplerBodyGenImpl(SPSTNode *Root)
{
  if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(Root)) {
    // Random Interleaving model will first collect the intra-thread RI, which
    // is independent with the interleaving type. To improve the performance, we
    // use UNIFORM_INTERLEAVING
    //
    // Uniform Interleaving model instead use sequential RI, so it will treat all
    // loop sequential.
    if (LoopNode->isParallelLoop() && InterleavingTechnique == 1) {
      CodeGen->EmitComment("Generate parallel code for " +
                           LoopNode->getLoopStringExpr());
      ParallelSamplerBodyGenImpl(LoopNode, Graphs[LoopNode], UNIFORM_INTERLEAVING);
    } else {
      string forloop = EmitLoopNodeExpr(LoopNode);
      CodeGen->EmitCode(forloop + " {");
      CodeGen->tab_count++;
      for (auto neighbor : LoopNode->neighbors) {
        SamplerBodyGenImpl(neighbor);
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
    }
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Root)) {
    string branch = EmitBranchCondExpr(Branch);
    // generate the code if condition is true
    if (!Branch->neighbors[0]->neighbors.empty()) {
      CodeGen->EmitCode(branch + " {");
      CodeGen->tab_count++;
      SamplerBodyGenImpl(Branch->neighbors[0]);
      CodeGen->tab_count--;
    }
    // we add this branch to avoid false branch is empty
    if (!Branch->neighbors[1]->neighbors.empty()) {
      if (Branch->neighbors[0]->neighbors.empty()) {
        CodeGen->EmitCode(branch + " {");
      } else {
        CodeGen->EmitCode("} else {");
      }
      CodeGen->tab_count++;
      // generate the code if condition is false
      SamplerBodyGenImpl(Branch->neighbors[1]);
      CodeGen->tab_count--;
    }
    CodeGen->EmitCode("}");
  } else if (RefTNode *RefNode = dynamic_cast<RefTNode *>(Root)) {
    stringstream ss;
    ss << "Generate address check func call for " << RefNode->getRefExprString();
    CodeGen->EmitComment(ss.str());
    ss.str("");
    vector<string> params = EmitRefNodeAccessExpr(RefNode, false);
    if (!params.empty()) {
      if (EnableSampling && SamplingMethod == SEQUENTIAL_START) {
        CodeGen->EmitComment("choose a number between 1 and 100");
        CodeGen->EmitStmt("int enable = rand() % 100 + 1");
        CodeGen->EmitCode("if (enable <= " + to_string(SamplingRatio) + ") {");
        CodeGen->tab_count++;
      }
//      CodeGen->EmitStmt("subscripts = " + params[1]);
      ss << "addr = GetAddress_" << SPSNodeNameTable[RefNode] << "(";
      for (unsigned i = 0; i < RefNode->getSubscripts().size(); i++) {
//        ss << "subscripts[" << i << "]";
        ss << EmitRefNodeAccessExprAtIdx(RefNode->getSubscripts()[i], false);
        if (i < RefNode->getSubscripts().size()-1)
          ss << ",";
      }
      ss << ")";
      CodeGen->EmitStmt(ss.str());
      ss.str("");
      AccPath path;
      QueryEngine->GetPath(RefNode, nullptr, path);
      ss << "access = new Iteration(\"" << SPSNodeNameTable[RefNode] << "\", {";
      auto pathIter = path.rbegin();
      TranslateStatus status = SUCCESS;
      while (pathIter != path.rend()) {
        LoopTNode *NestLoop = dynamic_cast<LoopTNode *>(*pathIter);
        assert(NestLoop && "All node in a path should be an instance of LoopTNode");
        string nest_loop_induction = ReplaceInductionVarExprByType(NestLoop->getInductionPhi(), SAMPLE, status);
        ss << nest_loop_induction;
        if (pathIter + 1 != path.rend())
          ss << ",";
        pathIter++;
      }
      ss << "})";
      CodeGen->EmitDebugInfo(ss.str());
      ss.str("");
      if (InterleavingTechnique == 0) {
        CodeGen->EmitCode("if (LAT_" + RefNode->getArrayNameString() +
                          ".find(addr) != LAT_" +
                          RefNode->getArrayNameString() + ".end()) {");
      } else {
        // THREAD_NUM+1 tracks all RIs in the sequential region
        // These RIs will not be passed to the model
        CodeGen->EmitCode("if (LAT_" + RefNode->getArrayNameString() +
                          "[THREAD_NUM+1].find(addr) != LAT_" +
                          RefNode->getArrayNameString() + "[THREAD_NUM+1].end()) {");
      }
      CodeGen->tab_count++;
      if (InterleavingTechnique == 0) {
        CodeGen->EmitStmt("long reuse = count - LAT_" +
                          RefNode->getArrayNameString() +
                          "[addr]");
      } else {
        CodeGen->EmitStmt("long reuse = count[THREAD_NUM+1] - LAT_" +
                          RefNode->getArrayNameString() +
                          "[THREAD_NUM+1][addr]");
      }
      CodeGen->EmitHashMacro("#if defined(DEBUG)");
      if (InterleavingTechnique == 0) {
        CodeGen->EmitStmt("Iteration *src = LATIterMap[LAT_" +
                          RefNode->getArrayNameString() +
                          "[addr]]");
      } else {
        CodeGen->EmitStmt("Iteration *src = LATIterMap["
                          "THREAD_NUM+1][LAT_" +
                          RefNode->getArrayNameString() +
                          "[THREAD_NUM+1][addr]]");
      }
#if 0
      //      CodeGen->EmitCode("if (reuse > 0.9*(" + EmitOneParallelIterationTripFormula(QueryEngine->GetParallelLoopDominator(RefNode)) + ")) {");
      CodeGen->EmitCode("if (distance_to(reuse,0) > distance_to(reuse," + parallel_iteration_cnt + ")) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("cout << \"[\" << reuse << \"] \" << src->toString() << \" -> \" << access->toString() << endl");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");

#endif
      CodeGen->EmitHashMacro("#endif");
      CodeGen->EmitFunctionCall("pluss_histogram_update", {"reuse", "1."});
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      if (InterleavingTechnique == 0) {
        CodeGen->EmitStmt("LAT_"+RefNode->getArrayNameString()+"[addr] = count");
        CodeGen->EmitDebugInfo("LATIterMap[count] = access");
        CodeGen->EmitStmt("count += 1");
      } else {
        CodeGen->EmitStmt("LAT_"+RefNode->getArrayNameString()+"[THREAD_NUM+1][addr] = count[THREAD_NUM+1]");
        CodeGen->EmitDebugInfo("LATIterMap[THREAD_NUM+1][count[THREAD_NUM+1]] = access");
        CodeGen->EmitStmt("count[THREAD_NUM+1] += 1");
      }
//      CodeGen->EmitFunctionCall("pluss_ref_access", {params[0],"addr"});
      if (EnableSampling) {
        if (SamplingMethod == SEQUENTIAL_START || SamplingMethod == BURSTY) {
          CodeGen->tab_count--;
          CodeGen->EmitCode("}");
        }
      }
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Root)) {
    for (auto neighbor : Dummy->neighbors) {
      SamplerBodyGenImpl(neighbor);
    }
  }
}


/* Generate the loop iteration vector update func */
void ModelCodeGenProWrapperPass::LoopIterUpdateGen(AccGraph *G,
                                                   AccGraphEdge *E,
                                                   LoopTNode* IDomLoop,
                                                   bool EnableIterationInc,
                                                   bool EnableIterationUpdate)
{
  // increment part
  stringstream codess;
  TranslateStatus status = SUCCESS;
  string induction_expr = Translator->ValueToStringExpr(IDomLoop->getInductionPhi(),
                                                        status);
  string loop_init_expr = induction_expr + "_lb",
      induction_comp_predicate = "<",
      loop_final_expr = induction_expr + "_ub",
      loop_step = induction_expr + "++";
  if (EnableIterationInc) {
    LoopBound *IDomLB = IDomLoop->getLoopBound();
    if (IDomLB) {
      Translator->PredicateToStringExpr(IDomLB->Predicate, status);
      if (status == SUCCESS)
        induction_comp_predicate = Translator->PredicateToStringExpr(IDomLB->Predicate, status);
      if (IDomLB->FinalValue) {
        ReplaceInductionVarExprByType(IDomLB->FinalValue, OMP_PARALLEL, status);
        if (status == SUCCESS)
          loop_final_expr =
              ReplaceInductionVarExprByType(IDomLB->FinalValue, OMP_PARALLEL, status);
      }
      ReplaceInductionVarExprByType(IDomLB->StepInst, OMP_PARALLEL, status);
      if (status == SUCCESS)
        loop_step =
            ReplaceInductionVarExprByType(IDomLB->StepInst, OMP_PARALLEL, status);
    }  else {
      LLVM_DEBUG(dbgs() << "No parsable Loop Bound. "
                           "Sampler will use the default setting\n");
      status = NOT_TRANSLATEABLE;
    }
    // distance measures the number of loops between IDomLoop to the
    // parallel loop. 0 means IDomLoop is the parallel loop
    int distance = ComputeDistanceToParallelLoop(IDomLoop);
    assert(distance >= 0 && "Immediate Loop dominator should be inside a "
                            "parallel loop nest");
    string progressRepr = "progress[tid_to_run]->iteration[" + to_string(distance) + "]";
    // IDomLoop is the parallel loop
    if (distance == 0) {
      codess << progressRepr << " = " << loop_step;
      CodeGen->EmitStmt(codess.str());
      codess.str("");
      CodeGen->EmitCode("if (progress[tid_to_run]->isInBound()) {");
      CodeGen->tab_count++;
    } else {
      if (IDomLB) {
        if (IDomLB->isLHS())
          codess << "if (" << loop_step << induction_comp_predicate
                 << loop_final_expr << ") {";
        else
          codess << "if (" << loop_final_expr << induction_comp_predicate
                 << loop_step << ") {";
      }
      CodeGen->EmitCode(codess.str());
      codess.str("");
      CodeGen->tab_count++;
      codess << progressRepr << " = " << loop_step;
      CodeGen->EmitStmt(codess.str());
      codess.str("");
//      CodeGen->tab_count--;
//      CodeGen->EmitCode("}");
    }
  }
  if (EnableIterationUpdate) {
    // update part
    // pop iteration vector L1.size() times
    // push LB of loop in L2 in iteration vector
    AccPath L1, L2;
    auto loopIter = L1.begin();
    auto loopRIter = L2.rbegin();
    G->QueryEngine->GetPath(E->getSrc(), IDomLoop, L1);
    G->QueryEngine->GetPath(E->getSink(), IDomLoop, L2);
    for (loopIter = L1.begin(); loopIter != L1.end(); ++loopIter) {
      CodeGen->EmitStmt("progress[tid_to_run]->iteration.pop_back()");
    }
    codess.str("");
    for (loopRIter = L2.rbegin(); loopRIter != L2.rend(); ++loopRIter) {
      LoopTNode *NodeInL2 = dynamic_cast<LoopTNode *>(*loopRIter);
      assert(NodeInL2 && "Nodes in path should be a LoopTNode object");
      LoopBound *NodeInL2LB = NodeInL2->getLoopBound();
      if (NodeInL2LB) {
        ReplaceInductionVarExprByType(NodeInL2LB->InitValue, OMP_PARALLEL, status);
        if (status == SUCCESS)
          loop_init_expr =
              ReplaceInductionVarExprByType(NodeInL2LB->InitValue, OMP_PARALLEL, status);
      }  else {
        LLVM_DEBUG(dbgs() << "No parsable Loop Bound. "
                             "Sampler will use the default setting\n");
        status = NOT_TRANSLATEABLE;
      }
      codess << "progress[tid_to_run]->iteration.emplace_back(" << loop_init_expr << ")";
      CodeGen->EmitStmt(codess.str());
      codess.str("");
    }
  }
//		if (EnableIterationInc) {
//			errs() << space << "    continue; /* go back to the interleaving */ \n";
//			errs() << space << "}\n";
//		}

  return;
}

void ModelCodeGenProWrapperPass::PerRefSamplerBodyGenImpl(SPSTNode *Root,
                                                          RefTNode *TargetRef)
{
  if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(Root)) {
    // If LoopNode does not contains any access to the TargetRef
    // we increment the counter with loop trip * access times
    // i.e.  count += loop_access_count
    // otherwise, we recursively generate sampler code for its children
    AccPath dominators;
    QueryEngine->GetPath(TargetRef, nullptr, dominators);
    if (QueryEngine->isReachable(LoopNode, TargetRef) && find(dominators.begin(), dominators.end(), LoopNode) != dominators.end()) {
      // when met a parallel loop node, parallel sampler body will be generated
      // if and only if parallel model is not work or modelopt option is not enabled
      AccPath LoopDominatorsOfTarget;
      QueryEngine->GetPath(TargetRef, nullptr, LoopDominatorsOfTarget);
      if (find(LoopDominatorsOfTarget.begin(), LoopDominatorsOfTarget.end(), LoopNode) ==
          LoopDominatorsOfTarget.end()) {
        CodeGen->EmitCode("if (start_reuse_search) {");
        CodeGen->tab_count++;
      }
      if (LoopNode->isParallelLoop() &&
          !(EnableModelOpt && (InterleavingTechnique == 0) && ModelValidateLoops.find(LoopNode) != ModelValidateLoops.end())) {
        CodeGen->EmitComment("Generate parallel code for " +
                             LoopNode->getLoopStringExpr());
        if (InterleavingTechnique == 0)
          (
              LoopNode, TargetRef, Graphs[LoopNode], UNIFORM_INTERLEAVING);
        else
          PerRefParallelSamplerBodyGenImpl(
              LoopNode, TargetRef, Graphs[LoopNode], UNIFORM_INTERLEAVING);
        //        PerPerRefParallelSamplerBodyGenImplRefParallelSamplerBodyGenImpl(LoopNode, TargetRef, Graphs[LoopNode]);
        // ParallelSamplerBodyGenImpl(LoopNode, Graphs[LoopNode]);
      } else {
        // if loop is not parallel loop or modelopt is enable and is valid
        AccPath path;
        QueryEngine->GetPath(TargetRef, nullptr, path);
        if (find(path.begin(), path.end(), LoopNode) == path.end()) {

          string forloop = EmitLoopNodeExpr(LoopNode);
          CodeGen->EmitCode(forloop + " {");
          CodeGen->tab_count++;
        } else {
          if (EnableSampling && SamplingMethod == RANDOM_START) {
            /**
            * if (ci == sample_ci) { ci+1_start  = sample_ci+1; }
            * for (ci+1 = ci+1_start; ci+1 < ci_ub; ci+1++)
             */
            string induction_expr =
                InductionVarNameTable[LoopNode->getInductionPhi()];
            if (LoopNode == *(path.rbegin())) {
              string sample_start_expr = "sample_" + induction_expr + "_start";
              CodeGen->EmitStmt(induction_expr + "_start=" + sample_start_expr);
            } else {
              stringstream ss;
              ss << "if (";
              AccPath dominators;
              QueryEngine->GetPath(LoopNode, nullptr, dominators);
              auto dom_riter = dominators.rbegin();
              while (dom_riter != dominators.rend()) {
                LoopTNode *dominator = dynamic_cast<LoopTNode *>(*dom_riter);
                assert(dominator &&
                           "The immediaate loop dominator of a loop inside a"
                           "loop nest must exist");
                string dom_induction_expr =
                    InductionVarNameTable[dominator->getInductionPhi()];
                ss << dom_induction_expr << "==sample_" << dom_induction_expr
                   << "_start";
                if (*dom_riter != dominators.front())
                  ss << " && ";
                dom_riter++;
              }
              ss << ") {";
              CodeGen->EmitCode(ss.str());
              ss.str("");
              CodeGen->tab_count++;
              string sample_start_expr = "sample_" + induction_expr + "_start";
              CodeGen->EmitStmt(induction_expr + "_start=" + sample_start_expr);
              CodeGen->tab_count--;
              CodeGen->EmitCode("} else {");
              CodeGen->tab_count++;
              string loop_init_expr = induction_expr + "_lb";
              TranslateStatus status = SUCCESS;
              Translator->ValueToStringExpr(LoopNode->getLoopBound()->InitValue,
                                            status);
              if (status == SUCCESS)
                loop_init_expr = Translator->ValueToStringExpr(
                    LoopNode->getLoopBound()->InitValue, status);
              CodeGen->EmitStmt(induction_expr + "_start=" + loop_init_expr);
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
            }
          }
          // generate the start of the loop
          string forloop = EmitLoopNodeExpr(LoopNode, EnableSampling && SamplingMethod == RANDOM_START);
          CodeGen->EmitCode(forloop + " {");
          CodeGen->tab_count++;
        }
        for (auto neighbor : LoopNode->neighbors) {
          PerRefSamplerBodyGenImpl(neighbor, TargetRef);
        }
        CodeGen->tab_count--;
        CodeGen->EmitCode("}"); // the end of the loop
      }
      if (find(LoopDominatorsOfTarget.begin(), LoopDominatorsOfTarget.end(), LoopNode) ==
          LoopDominatorsOfTarget.end()) {
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
      }
    } else {
      // we need to check if there is any iv dependencies or there are branches
      // inside the loop, if so, we need to generate this loop.
      // otherwise, since all loop bounds are constant, we can generate a
      // representation that computes the number of accesses in this loop
      if (!canBeSimplifiedByFormula(LoopNode)) {
        LLVM_DEBUG(dbgs() << "Cannot generate the trip formula for "
                          << LoopNode->getLoopStringExpr() << "\n");
        // Before generating code for its children, we need to generate the
        // code for this LoopNode first
        string forloop = EmitLoopNodeExpr(LoopNode);
        CodeGen->EmitCode(forloop + " {");
        CodeGen->tab_count++;
        for (auto neighbor : LoopNode->neighbors) {
          PerRefSamplerBodyGenImpl(neighbor, TargetRef);
        }
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
      } else {
        // generate and emit the trip count expression of this loop
        stringstream ss;
        ss << LoopNode->getLoopStringExpr() << " does not have references that";
        ss << " access " << TargetRef->getArrayNameString();
        CodeGen->EmitComment(ss.str());
        string loop_acc_count_expr = EmitTripFormula(LoopNode);
        // currently, we do not disable the bypass oxcept random-start sampling
#if 0

        if (EnableSampling && SamplingMethod == BURSTY) {
          CodeGen->EmitStmt("enable = !enable && (access_cnt % "
                            + to_string(SamplingInterval) + " == 0)");
          CodeGen->EmitStmt("access_cnt += " + loop_acc_count_expr);
          CodeGen->EmitCode("if (enable) {");
          CodeGen->tab_count++;
        } else
#endif
        CodeGen->EmitCode("if (start_reuse_search) {");
        CodeGen->tab_count++;
        if (InterleavingTechnique == 0)
          CodeGen->EmitStmt("count += (long)" + loop_acc_count_expr);
        else
          CodeGen->EmitStmt("count[tid_to_run] += (long)(" + loop_acc_count_expr + "/ THREAD_NUM)");
        CodeGen->tab_count--;
        CodeGen->EmitCode("} // end of start_reuse_search");
      }
    }
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Root)) {
    // If neither true or false Branch contains any access to the TargetRef
    // we increment the counter with the trip count of each branches but we still
    // generate the branch statmenet
    // i.e.
    // if (cond) {
    //    count += true_access_count
    // } else {
    //    count += false_access_count
    // }
    // otherwise, we only generate the path that has the access to the root

    // a special case is that, the compiler pass will reverse the if condition
    // making the origin code to
    // if (cond) {
    // } else {
    //  false path
    // }
    // in this case, there is no neighbors in true branch and we will reverse
    // the predicate of the condition
    // if (!cond) {
    //  flase path
    // }
    string branch = EmitBranchCondExpr(Branch);
    if (!Branch->neighbors[0]->neighbors.empty()) {
      CodeGen->EmitCode(branch + " {");
      CodeGen->tab_count++;
      bool isReachableFromTrueBranch = QueryEngine->isReachable(Branch->neighbors[0], TargetRef);
      if (isReachableFromTrueBranch || !canBeSimplifiedByFormula(Branch->neighbors[0])) {
        // generate the code if condition is true
        PerRefSamplerBodyGenImpl(Branch->neighbors[0], TargetRef);
//      SamplerBodyGenImpl(Branch->neighbors[0]);
      } else {
        // generate and emit the trip count expression of the true branch
        string true_branch_acc_count_expr = EmitTripFormula(Branch->neighbors[0]);
#if 0
        if (EnableSampling && SamplingMethod == BURSTY)
        CodeGen->EmitStmt("access_cnt += " + true_branch_acc_count_expr);
#endif
        CodeGen->EmitCode("if (start_reuse_search) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("count += (long)" + true_branch_acc_count_expr);
//        if (EnableParallelOpt) {
//          CodeGen->EmitStmt("m.lock()");
//          CodeGen->EmitFunctionCall("pluss_per_thread_bypass",
//                                    {"this_thread::get_id()", true_branch_acc_count_expr});
//          CodeGen->EmitStmt("m.unlock()");
//        } else {
//          CodeGen->EmitFunctionCall("pluss_bypass", {true_branch_acc_count_expr});
//        }
        CodeGen->tab_count--;
        CodeGen->EmitCode("} // end of start_reuse_search");
      }
      CodeGen->tab_count--;
    }

    // we add this branch to avoid false branch is empty
    if (!Branch->neighbors[1]->neighbors.empty()) {
      if (Branch->neighbors[0]->neighbors.empty()) {
        string branch = EmitBranchCondExpr(Branch);
        CodeGen->EmitCode(branch + " {");
      } else {
        CodeGen->EmitCode("} else {");
      }
      CodeGen->tab_count++;
      bool isReachableFromFalseBranch = QueryEngine->isReachable(Branch->neighbors[1], TargetRef);
      if (isReachableFromFalseBranch || !canBeSimplifiedByFormula(Branch->neighbors[1])) {
        // generate the code if condition is false
        PerRefSamplerBodyGenImpl(Branch->neighbors[1], TargetRef);
      } else {
        // generate and emit the trip count expression of the false branch
        string false_branch_acc_count_expr = EmitTripFormula(Branch->neighbors[1]);
#if 0
        if (EnableSampling && SamplingMethod == BURSTY)
          CodeGen->EmitStmt("access_cnt += " + false_branch_acc_count_expr);
#endif
        CodeGen->EmitCode("if (start_reuse_search) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("count += (long)" + false_branch_acc_count_expr);

        CodeGen->tab_count--;
        CodeGen->EmitCode("} // end of start_reuse_search");
      }
      CodeGen->tab_count--;
    }
    CodeGen->EmitCode("}");
  } else if (RefTNode *RefNode = dynamic_cast<RefTNode *>(Root)) {
    // If RefNode access the same array as the TargetRef and it is reachable from
    // the TargetRef, then we generate the address check call
    // otherwise, we increment the counter by one
    stringstream ss;
    if (QueryEngine->areAccessToSameArray(RefNode, TargetRef) && QueryEngine->GetParallelLoopDominator(RefNode) == QueryEngine->GetParallelLoopDominator(TargetRef)) {
      ss << "Generate address check func call for " << RefNode->getRefExprString();
      ss << "(" << SPSNodeNameTable[RefNode] << ")";
      CodeGen->EmitComment(ss.str());
      ss.str("");
      bool parallel_model_is_generated = false;
      if (RefNode == TargetRef) {
        CodeGen->EmitCode("if (!start_reuse_search) { start_reuse_search=true; }");
      }
      CodeGen->EmitCode("if (start_reuse_search) {");
      CodeGen->tab_count++;
      vector<string> params = EmitRefNodeAccessExpr(RefNode, false);
      if (!params.empty()) {
        AccPath path;
        QueryEngine->GetPath(RefNode, nullptr, path);
        if (path.empty() || SPSNodeNameTable[RefNode] != SPSNodeNameTable[TargetRef]) {
          // special case, the RefNode is not in a loop
//          ss << "Iteration *access = nullptr";
        } else {
          ss << "Iteration *access = new Iteration(\"" << SPSNodeNameTable[RefNode] << "\", {";
          auto pathIter = path.rbegin();
          TranslateStatus status = SUCCESS;
          while (pathIter != path.rend()) {
            LoopTNode *NestLoop = dynamic_cast<LoopTNode *>(*pathIter);
            assert(NestLoop && "All node in a path should be an instance of LoopTNode");
            string nest_loop_induction = ReplaceInductionVarExprByType(NestLoop->getInductionPhi(), SAMPLE, status);
            ss << nest_loop_induction;
            if (pathIter + 1 != path.rend())
              ss << ",";
            pathIter++;
          }
          ss << "})";
          CodeGen->EmitStmt(ss.str());
          ss.str("");
        }
//        if (EnableSampling) {
//          CodeGen->EmitFunctionCall("pluss_sample_access_parallel", params);
//        } else {
//          CodeGen->EmitFunctionCall("pluss_sample_access", params);
//        }
        CodeGen->EmitStmt("string array = " + params[0]);
        CodeGen->EmitStmt("string refname = \"" + SPSNodeNameTable[RefNode] + "\"");
//        CodeGen->EmitStmt("subscripts = " + params[1]);
        ss << "addr = GetAddress_" << SPSNodeNameTable[RefNode] << "(";
        for (unsigned i = 0; i < RefNode->getSubscripts().size(); i++) {
//          ss << "subscripts[" << i << "]";
          ss << EmitRefNodeAccessExprAtIdx(RefNode->getSubscripts()[i], false);
          if (i < RefNode->getSubscripts().size()-1)
            ss << ",";
        }
        ss << ")";
        CodeGen->EmitStmt(ss.str());
        ss.str("");
        if (EnableSampling && SamplingMethod == RANDOM_START) {
          CodeGen->EmitStmt("bool isSample = false");
          if (RefNode == TargetRef) {
            CodeGen->EmitDebugInfo("cout << access->toString() << \" @ \" << addr << endl");
            CodeGen->EmitCode(
                "if (access->compare(start_iteration) == 0) {");
            CodeGen->tab_count++;
            CodeGen->EmitDebugInfo("cout << \"Meet the start sample \" << access->toString() << endl");
            CodeGen->EmitStmt("isSample = true");
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else if ((!samples.empty() && access->compare(samples.top()) == 0) "
                              "|| (sample_names.find(access->toAddrString()) != sample_names.end())) {");
            CodeGen->tab_count++;
            CodeGen->EmitDebugInfo(
                "cout << \"Meet a new sample \" << access->toString() << \" while searching reuses\" << endl");
            CodeGen->EmitStmt("traversing_samples++");
            CodeGen->EmitCode("if (!samples.empty() && access->compare(samples.top()) == 0) {");
            CodeGen->tab_count++;
            CodeGen->EmitStmt("samples.pop()");
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
            CodeGen->EmitStmt("isSample = true");
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
          }
        }
        CodeGen->EmitCode("if (LAT[tid_to_run].find(addr) != LAT[tid_to_run].end()) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("long reuse = count[tid_to_run] - LAT[tid_to_run][addr]");
        CodeGen->EmitStmt("Iteration *src = LATSampleIterMap[tid_to_run][LAT[tid_to_run][addr]]");
        // compute the expression of one iteration of parallel loop
        // replace all induction variable names inside the expression with
        // the progress->iteration vector.
        if (LoopTNode *ParallelLoopDominator = QueryEngine->GetParallelLoopDominator(RefNode)) {
          string parallel_iteration_cnt =
              EmitOneParallelIterationTripFormula(ParallelLoopDominator);
          AccPath PathToParallelLoop;
          QueryEngine->GetPath(RefNode, ParallelLoopDominator,
                               PathToParallelLoop);
          auto pathIter = PathToParallelLoop.rbegin();
          int distance = isInsideParallelLoopNest(
              ParallelLoopDominator->getInductionPhi());
          string to_replace =
              "progress[tid_to_run]->iteration[" + to_string(distance) + "]";
          ReplaceSubstringWith(
              parallel_iteration_cnt,
              InductionVarNameTable[ParallelLoopDominator->getInductionPhi()],
              to_replace);
          while (pathIter != PathToParallelLoop.rend()) {
            LoopTNode *NestLoop = dynamic_cast<LoopTNode *>(*pathIter);
            assert(NestLoop &&
                       "All node in a path should be an instance of LoopTNode");
            distance = isInsideParallelLoopNest(NestLoop->getInductionPhi());
            to_replace =
                "progress[tid_to_run]->iteration[" + to_string(distance) + "]";
            ReplaceSubstringWith(
                parallel_iteration_cnt,
                InductionVarNameTable[NestLoop->getInductionPhi()], to_replace);
            pathIter++;
          }
          switch (isInterThreadSharingRef(RefNode)) {
          case FULL_SHARE: {
            // sharing THREAD_NUM -1 elements
            CodeGen->EmitCode("if (distance_to(reuse,0) > distance_to(reuse," +
                              parallel_iteration_cnt + ")) {");
            CodeGen->tab_count++;
            CodeGen->EmitFunctionCall(
                "pluss_parallel_histogram_update",
                {"share_intra_thread_RI[THREAD_NUM-1]", "reuse", "1"});
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else {");
            CodeGen->tab_count++;
            CodeGen->EmitFunctionCall(
                "pluss_parallel_histogram_update",
                {"no_share_intra_thread_RI", "reuse", "1"});
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
            break;
          }
          case SPATIAL_SHARE: {
            // sharing CLS/(DS*C)
            CodeGen->EmitCode("if (CHUNK_SIZE < (CLS / DS)) {");
            CodeGen->tab_count++;
            CodeGen->EmitCode("if (distance_to(reuse,0) > distance_to(reuse," +
                              parallel_iteration_cnt + ")) {");
            CodeGen->tab_count++;
            CodeGen->EmitFunctionCall(
                "pluss_parallel_histogram_update",
                {"share_intra_thread_RI[CLS/(DS*CHUNK_SIZE)-1]", "reuse", "1."});
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else {");
            CodeGen->tab_count++;
            CodeGen->EmitFunctionCall(
                "pluss_parallel_histogram_update",
                {"no_share_intra_thread_RI", "reuse", "1."});
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else {");
            CodeGen->tab_count++;
            CodeGen->EmitFunctionCall(
                "pluss_parallel_histogram_update",
                {"no_share_intra_thread_RI", "reuse", "1."});
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
            break;
          }
          default:
            CodeGen->EmitFunctionCall(
                "pluss_parallel_histogram_update",
                {"no_share_intra_thread_RI", "reuse", "1."});
            break;
          }
        } else {
          CodeGen->EmitFunctionCall("pluss_histogram_update", {"reuse", "1."});
        }

        if (EnableSampling && SamplingMethod == RANDOM_START) {
#if 0
          CodeGen->EmitStmt("samples.erase(*src)");
#endif
          if (TargetRef == RefNode) {
            CodeGen->EmitDebugInfo(
                "cout << \"[\" << reuse << \"] \" << src->toString() << \" -> \" << access->toString() << endl");
          }
          CodeGen->EmitStmt("traversing_samples--");
          CodeGen->EmitComment("stop traversing if reuse of all samples"
                               "are found");
          CodeGen->EmitDebugInfo("if (samples.empty() && traversing_samples == 0) { cout << \"[\" << reuse << \"] for last sample \" << src->toString() << endl; }");
          CodeGen->EmitCode("if (samples.empty() && traversing_samples == 0) { goto END_SAMPLE; }");
          CodeGen->EmitCode("if (traversing_samples == 0) {");
          CodeGen->tab_count++;
          CodeGen->EmitDebugInfo("cout << \"delete sample \" << src->toString() << \", active:\" << traversing_samples << \", remain:\" << samples.size() << endl");
          CodeGen->EmitStmt("delete src");

          CodeGen->EmitStmt("LATSampleIterMap[tid_to_run].erase(LAT[tid_to_run][addr])");
          CodeGen->EmitStmt("LAT[tid_to_run].erase(addr)");

          CodeGen->EmitComment("Here we examine if there is an out-of-order effect.");
          CodeGen->EmitComment("if the next sample we should jump has been traversed before, we will pop this sample directly.");
          CodeGen->EmitComment("It is safe to call samples.top() once, since when entering here, 'samples' queue is not empty()");
          CodeGen->EmitCode("if (samples_meet.size() >= samples.size()) { goto END_SAMPLE; }");
          CodeGen->EmitStmt("Iteration next = samples.top()");
          CodeGen->EmitCode("while(samples_meet.find(next.toAddrString()) != samples_meet.end()) {");
          CodeGen->tab_count++;
          CodeGen->EmitDebugInfo("cout << \"Skip \" << next.toString() << \" because we met this sample already \" << endl");
          CodeGen->EmitStmt("samples.pop()");
          CodeGen->EmitComment("All samples has been traversed, no need to jump");
          CodeGen->EmitCode("if (samples.empty()) { break; }");
          CodeGen->EmitStmt("next = samples.top()");
          CodeGen->tab_count--;
          CodeGen->EmitCode("} // end of out-of-order check");
          CodeGen->EmitCode("if (!samples.empty()) {");
          CodeGen->tab_count++;
          CodeGen->EmitHashMacro("#if defined(DEBUG)");
          CodeGen->EmitStmt("next = samples.top()");
          CodeGen->EmitStmt("cout << \"Jump to next sample \" << next.toString() << endl");
          CodeGen->EmitHashMacro("#endif");
          CodeGen->EmitStmt("goto START_SAMPLE_"+SPSNodeNameTable[TargetRef]);
          CodeGen->tab_count--;
          CodeGen->EmitCode("} else { goto END_SAMPLE; }");
          CodeGen->tab_count--;
          CodeGen->EmitCode("} // end of if (traversing_samples == 0)");
        }
        CodeGen->EmitDebugInfo("cout << \"delete sample \" << src->toString() << \", active:\" << traversing_samples << \", remain:\" << samples.size() << endl");
        CodeGen->EmitStmt("delete src");

        CodeGen->EmitStmt("LATSampleIterMap[tid_to_run].erase(LAT[tid_to_run][addr])");
        CodeGen->EmitStmt("LAT[tid_to_run].erase(addr)");

        CodeGen->tab_count--;
        CodeGen->EmitCode("} // end of if (LAT.find(addr) != LAT.end())");
        if (RefNode == TargetRef) {
          if (EnableSampling && SamplingMethod == RANDOM_START) {
#if 0
            CodeGen->EmitCode(
                "if (samples.find(*access) != samples.end()) {");
#endif
            CodeGen->EmitCode("if (isSample) {");
            CodeGen->tab_count++;
            if (EnableModelOpt && InterleavingTechnique == 0) {
              CodeGen->EmitStmt("LATSampleIterMap[count] = access");
              CodeGen->EmitStmt("LAT[addr] = count");
              CodeGen->EmitStmt("sequentialLAT[access] = sequential_count");
            } else {
              CodeGen->EmitStmt("LATSampleIterMap[tid_to_run][count[tid_to_run]] = access");
              CodeGen->EmitStmt("LAT[tid_to_run][addr] = count[tid_to_run]");
            }
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else { delete access; }");
          } else {
            CodeGen->EmitStmt("LAT[addr] = count");
          }
        }
      }

      CodeGen->EmitStmt("count[tid_to_run] += 1");

      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of if (start_reuse_search)");
    } else {
      ss << RefNode->getRefExprString() << " will not access " << TargetRef->getArrayNameString();
      CodeGen->EmitComment(ss.str());
      ss.str("");
#if 0
      if (EnableSampling && SamplingMethod == BURSTY) {
        CodeGen->EmitStmt("enable = !enable && (access_cnt % "
                          + to_string(SamplingInterval) + " == 0)");
        CodeGen->EmitStmt("access_cnt += 1");
        CodeGen->EmitCode("if (enable) {");
        CodeGen->tab_count++;
      }
#endif
      CodeGen->EmitCode("if (start_reuse_search) {");
      CodeGen->tab_count++;
      if (InterleavingTechnique == 0)
        CodeGen->EmitStmt("count += 1");
      else
        CodeGen->EmitStmt("count[tid_to_run] += 1");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Root)) {
    // dummy node are always considered reachable to the target
    // dummy node can be either 1) root or 2) one condition branch
    // the first case is always true for any reference and the reachability of
    // the second case had already been examined when we meet the branch node
    for (auto neighbor : Dummy->neighbors) {
      PerRefSamplerBodyGenImpl(neighbor, TargetRef);
    }
  }
}

void ModelCodeGenProWrapperPass::PerRefParallelSamplerBodyGenImpl(SPSTNode *Root,
                                                                  RefTNode *TargetRef, AccGraph *G, Interleaving type)
{
  // when entering this function, we guaranteed that Root is reachable to
  // the TargetRef (at least of one child of Root accessed the same array with
  // the TargetRef
  LoopTNode *ParallelLoopNode = dynamic_cast<LoopTNode *>(Root);
  AccPath pathToTreeRoot;
  QueryEngine->GetPath(ParallelLoopNode, nullptr, pathToTreeRoot);
  AccPath TargetToRoot;
  G->QueryEngine->GetPath(TargetRef, nullptr, TargetToRoot);
  unsigned DistanceToTop = 0; // 0 means the top loop is parallelized
  if (!pathToTreeRoot.empty()) { DistanceToTop = pathToTreeRoot.size(); }
  assert(ParallelLoopNode && "The given node to parallel should be a loop");
  TranslateStatus status = SUCCESS;
  vector<SPSTNode *> NodesInGraph;
  G->GetRefTNodeInTopologyOrder(Root, NodesInGraph);
  vector<SPSTNode *>::iterator GraphNodeIter = NodesInGraph.begin();
  SPSTNode *currentNode = nullptr, *nextNode = nullptr;
  bool reachTheLastAccessNode = false;
  SchedulingType scheduling = STATIC;
  // we need to check if there is any Induction variable dependence of this
  // parallel loop. If yes, we should use dynamic scheduling
  if (hasInductionVarDependenceChildren(ParallelLoopNode))
    scheduling = DYNAMIC;
  if (scheduling == DYNAMIC)
    CodeGen->EmitComment("Threads are scheduled using"
                         " dynamic scheduling");
  else
    CodeGen->EmitComment("Threads are scheduled using"
                         " static scheduling");
  if (type == RANDOM_INTERLEAVING)
    CodeGen->EmitComment("Threads are interleaved using"
                         " random interleaving");
  else
    CodeGen->EmitComment("Threads are interleaved using"
                         " uniform interleaving");
#if 0
  // use to track all branches unclosed
  stack<BranchTNode *> OpenBranches;
#endif
#if 0
  // GraphNodeIter either points to a reference node, or a branch node
  // Filtering out those RefNode that does not access the same array as the
  // TargetRef.
  unsigned skipped_trip = 0;
  string skipped_trip_expr = "";
  while (true) {
    currentNode = (*GraphNodeIter);
    // two break conditions, either an branch node, or
    if (dynamic_cast<BranchTNode *>(currentNode))
      break;
    if (RefTNode *RefNode = dynamic_cast<RefTNode *>(currentNode)) {
      if (QueryEngine->areAccessToSameArray(RefNode, TargetRef))
        break;
    }
    // neither two conditions are met, this is an RefNode that should be skipped
    // we increment th number of accesses inside this trip.
    skipped_trip+=1;
    skipped_trip_expr = skipped_trip_expr + " + " + to_string(skipped_trip);
    // if the refnode is a source of a backedge, we also increment the skipped_trip
    // with the trip of the carried loop.
    SmallVector<AccGraphEdge *, 8> edges;
    G->GetEdgesWithSource(currentNode, edges);
    for (auto edge : edges) {
      if (LoopTNode *CarryLoopNode = edge->getCarryLoop()) {
        skipped_trip_expr += EmitTripFormula(CarryLoopNode);
      }
    }
    GraphNodeIter++;
  }
#endif
  stringstream commentss;
  stringstream  codess;
  while (true) {
    currentNode = (*GraphNodeIter);
    AccPath pathToNode;
    G->GetPathToNode(currentNode, Root, pathToNode);
    if (GraphNodeIter == NodesInGraph.begin()) {
      //  generate the preparation work at the very beginning
      //  this will be generate if and only if currently we are in the root
      //  of the access graph

      // trip counts the number of iterations of the parallel loop
      // default we assume the loop predicate is <
      string trip = RepresentLoopTrip(ParallelLoopNode);
      codess << "dispatcher = ChunkDispatcher(CHUNK_SIZE,(" << trip << ")";
      if (ParallelLoopNode->getLoopBound()) {
        codess << ",";
        codess << Translator->ValueToStringExpr(
            ParallelLoopNode->getLoopBound()->InitValue, status);
        codess << ",";
        codess << Translator->ValueToStringExpr(
            ParallelLoopNode->getLoopBound()->StepValue, status);
      }
      codess << ")";
      CodeGen->EmitStmt(codess.str());
      codess.str("");
      // if sampling is enabled, we should first generate all sample before
      // the chunk engine. Note that the sample had alreaady be generated before
      // entering here. All we need to do is generate the start point of each
      // thread based on our sample
      //
      // sample is different in terms of interleaving
      // Unform Interleaving:
      //    the start point of each thread is the same
      //    we only have to randomly chose one iteration and the start point
      //    of other threads can be computed
      //
      // Random Interleaving:
      //    the start point of each thread is different.
      //    we assume that all threads should at least stays within the same
      //    chunk.
      //    We need to randomly chose one iteration, one for each thread.

      // sampling will start only when the parallel loop contains the TargetRef
      // otherwise, it should start from the very beginning.
      if (EnableSampling && SamplingMethod == RANDOM_START) {
        if (find(TargetToRoot.begin(), TargetToRoot.end(), ParallelLoopNode) !=
            TargetToRoot.end()) {
          CodeGen->EmitStmt("dispatcher.reset()");
          CodeGen->EmitStmt("idle_threads.clear()");
          CodeGen->EmitStmt("worker_threads.clear()");
          CodeGen->EmitCode("for (tid_to_run = 0; tid_to_run < THREAD_NUM; tid_to_run++) {");
          CodeGen->tab_count++;
          CodeGen->EmitDebugInfo("cout << \"Move \" << tid_to_run << \" to idle_threads\" << endl");
          CodeGen->EmitStmt("idle_threads.emplace_back(tid_to_run)");
          CodeGen->tab_count--;
          CodeGen->EmitCode("}");
          // check whether we should start from the sampling iteration
          codess << "bool start_parallel_sample = ";
          if (pathToTreeRoot.empty()) {
            codess << "true";
          } else {
            codess << "(";
            auto toppath_riter = pathToTreeRoot.rbegin();
            while (toppath_riter != pathToTreeRoot.rend()) {
              LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*toppath_riter);
              if (*toppath_riter == ParallelLoopNode)
                break;
              string induction_expr = Translator->ValueToStringExpr(
                  LoopInPath->getInductionPhi(), status);
              codess << "(" << induction_expr << "==" << induction_expr
                     << "_start)";
              if ((toppath_riter != pathToTreeRoot.rend() - 1) &&
                  *(toppath_riter + 1) != ParallelLoopNode) {
                codess << " && ";
              }
              toppath_riter++;
            }
            codess << ")";
          }
          CodeGen->EmitStmt(codess.str());
          codess.str("");
#if 0
          CodeGen->EmitStmt(
              "unordered_map<int, Progress> per_thread_start_point");
#endif
          CodeGen->EmitCode("if (start_parallel_sample) {");
          CodeGen->tab_count++;
          // here we generate the code that computes the start point
          ParallelSamplerSamplingHeaderGen(ParallelLoopNode, G, TargetRef, type,
                                           scheduling);
#if 0
          CodeGen->EmitStmt("dispatcher.setStartPoint(start_iteration.ivs[" +
                          to_string(DistanceToTop) + "])");
#endif
          CodeGen->EmitStmt("start_parallel_sample = false");
          CodeGen->EmitComment("assign the start sample to each thread");
          CodeGen->EmitCode("if (!worker_threads.empty()) {");
          CodeGen->tab_count++;
          CodeGen->EmitStmt("goto INTERLEAVING_LOOP_"+to_string(ParallelLoopNode->getLoopID()));
          CodeGen->tab_count--;
          CodeGen->EmitCode("} else {");
          CodeGen->tab_count++;
          CodeGen->EmitStmt("goto END_SAMPLE");
          CodeGen->tab_count--;
          CodeGen->EmitCode("}");
          CodeGen->tab_count--;
          CodeGen->EmitCode("} // end of if (start_parallel_sample)");
        } else {
          CodeGen->EmitCode(
              "for (tid_to_run = 0; tid_to_run < THREAD_NUM; tid_to_run++) {");
          CodeGen->tab_count++;
          CodeGen->EmitDebugInfo("cout << \"Move \" << tid_to_run << \" to idle_threads\" << endl");
          CodeGen->EmitStmt("idle_threads.emplace_back(tid_to_run)");
          CodeGen->tab_count--;
          CodeGen->EmitCode("}");
        }
      } else {
        CodeGen->EmitCode(
            "for (tid_to_run = 0; tid_to_run < THREAD_NUM; tid_to_run++) {");
        CodeGen->tab_count++;
        CodeGen->EmitDebugInfo("cout << \"Move \" << tid_to_run << \" to idle_threads\" << endl");
        CodeGen->EmitStmt("idle_threads.emplace_back(tid_to_run)");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
      }

      CodeGen->EmitCode("while(true) {");
      if (type == RANDOM_INTERLEAVING)
        CodeGen->EmitLabel("CHUNK_ASSIGNMENT_LOOP_" +
                           to_string(ParallelLoopNode->getLoopID()));
      CodeGen->tab_count++;
      if (EnableSampling && SamplingMethod == RANDOM_START) {
        if (find(TargetToRoot.begin(), TargetToRoot.end(), ParallelLoopNode) !=
            TargetToRoot.end()) {
#if 0
          CodeGen->EmitCode("if (!per_thread_start_point.empty()) {");
          CodeGen->EmitStmt("auto idle_thread_iterator = idle_threads.begin()");
          CodeGen->EmitCode(
              "while (idle_thread_iterator != idle_threads.end()) {");
          CodeGen->tab_count++;
          CodeGen->EmitStmt("int idle_tid = *idle_thread_iterator");
          CodeGen->EmitCode("if (per_thread_start_point.find(idle_tid) == per_thread_start_point.end()) {");
          CodeGen->tab_count++;
          CodeGen->EmitStmt("idle_thread_iterator++");
          CodeGen->EmitStmt("continue");
          CodeGen->tab_count--;
          CodeGen->EmitCode("}");
          CodeGen->EmitStmt(
              "progress[idle_tid] = per_thread_start_point[idle_tid]");
          CodeGen->EmitStmt(
              "idle_thread_iterator = idle_threads.erase("
              "find(idle_threads.begin(), idle_threads.end(), idle_tid))");
          CodeGen->EmitDebugInfo("cout << \"Move \" << idle_tid << \" to worker_threads\" << endl");
          CodeGen->EmitStmt("worker_threads.push_back(idle_tid)");
          CodeGen->tab_count--;
          CodeGen->EmitCode("}");
          CodeGen->EmitStmt("per_thread_start_point.clear()");
          CodeGen->tab_count--;
          CodeGen->EmitCode(
              "} else if (!idle_threads.empty() && dispatcher.hasNextChunk(" +
              to_string(scheduling == STATIC) + ")) {");
#endif
          CodeGen->EmitCode(
              "if (!idle_threads.empty() && dispatcher.hasNextChunk(" +
              to_string(scheduling == STATIC) + ")) {");
        } else {
          CodeGen->EmitCode(
              "if (!idle_threads.empty() && dispatcher.hasNextChunk(" +
              to_string(scheduling == STATIC) + ")) {");
        }
      } else {
        CodeGen->EmitCode(
            "if (!idle_threads.empty() && dispatcher.hasNextChunk(" +
            to_string(scheduling == STATIC) + ")) {");
      }
      CodeGen->tab_count++;
      if (scheduling == STATIC) {
        CodeGen->EmitStmt("auto idle_threads_iterator = idle_threads.begin()");
        CodeGen->EmitCode("while(idle_threads_iterator != idle_threads.end() && dispatcher.hasNextChunk(" +
                          to_string(scheduling == STATIC) + ")) {");
      } else {
        CodeGen->EmitComment(
            "Randomly order the element in the idle_threads list. "
            "Chunks are assigned to threads in random order");
        CodeGen->EmitStmt(
            "random_shuffle(idle_threads.begin(), "
            "idle_threads.end(), [](int n) { return rand() % n; })");
        CodeGen->EmitCode(
            "while(!idle_threads.empty() && dispatcher.hasNextChunk(" +
            to_string(scheduling == STATIC) + ")) {");
      }
      CodeGen->tab_count++;
      if (scheduling == STATIC) {
        CodeGen->EmitStmt("tid_to_run = *idle_threads_iterator");
        CodeGen->EmitCode("if (!dispatcher.hasNextStaticChunk(tid_to_run)) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt("idle_threads_iterator++");
        CodeGen->EmitStmt("continue");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
      } else {
        CodeGen->EmitStmt("tid_to_run = *(idle_threads.begin())");
      }
      if (scheduling == STATIC)
        CodeGen->EmitStmt(
            "Chunk c = dispatcher.getNextStaticChunk(tid_to_run)");
      else
        CodeGen->EmitStmt("Chunk c = dispatcher.getNextChunk(tid_to_run)");
      CodeGen->EmitStmt("vector<int> parallel_iteration_vector");
      // we need the path to the root of the tree, then the path may includes
      // those loops that are the parent of the parallel loop
      AccPath pathToParallelLoop;
      G->GetPathToNode(currentNode, ParallelLoopNode, pathToParallelLoop);
      auto path_riter = pathToParallelLoop.rbegin();
      while (path_riter != pathToParallelLoop.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode type");
        //        LLVM_DEBUG(dbgs() << LoopInPath->getLoopStringExpr() << "\n");
        path_riter++;
      }

      // traverse the loop from outer to inner
      // if the loop is inside the parallel loop, we assign it with the
      // lower bound of the loop induction variable
      // if the loop is outside the parallel loop, we do not put it in the progress iteration vector. They can be viewed as a constant i.e. for (c0 = 0; c0 < n; c0 ++) #pragma omp
      //   for (c1 = 0; c1 < n; c1++)
      //     for (c2 = 0; c2 < n; c2++)
      // the iteration vector would be {c.first, 0}
      path_riter = pathToParallelLoop.rbegin();
      unsigned i = 0;
      while (path_riter != pathToParallelLoop.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode type");
        codess.str(""); // clear the stream just in case it is not empty
        if (LoopInPath == ParallelLoopNode) { // this is the parallel loop
          codess << "parallel_iteration_vector.push_back(c.first)";
          CodeGen->EmitStmt(codess.str());
          codess.str("");
        } else if (LoopInPath->getLoopLevel() <
                   ParallelLoopNode->getLoopLevel()) { // this is the parent of the parallel loop
          codess << Translator->ValueToStringExpr(LoopInPath->getInductionPhi(),
                                                  status);
        } else { // this is the loop inside the parallel loop
#if 0
          if (find(TargetToRoot.begin(), TargetToRoot.end(), ParallelLoopNode)
                                         != TargetToRoot.end()) {
            // only the parallel loop who is the parent of the TargetRef can
            // start from the sampled iteration.
            codess << "if (parallel_iteration_vector[" << (i-1) << "] == "
                   << "start_iteration.ivs[" << (i+DistanceToTop) << "]) {";
            CodeGen->EmitCode(codess.str());
            codess.str("");
            CodeGen->tab_count++;
            codess << "parallel_iteration_vector.push_back(start_iteration.ivs["
                << (i+DistanceToTop) << "])";
            CodeGen->EmitStmt(codess.str());
            codess.str("");
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else {");
            CodeGen->tab_count++;
            codess << "parallel_iteration_vector.push_back(" << ReplaceInductionVarExprByType(
                LoopInPath->getLoopBound()->InitValue, OMP_PARALLEL_INIT,
                status) << ")";
            CodeGen->EmitStmt(codess.str());
            codess.str("");
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
          } else {
#endif
          codess << "parallel_iteration_vector.push_back("
                 << ReplaceInductionVarExprByType(
                     LoopInPath->getLoopBound()->InitValue,
                     OMP_PARALLEL_INIT, status)
                 << ")";
          CodeGen->EmitStmt(codess.str());
          codess.str("");
#if 0
          }
#endif
        }
        codess.str("");
        path_riter++;
        i++;
      }
      //      SPSTNode *tmpNode = currentNode;
      //      while (ParentMappingInAbstractionTree.find(tmpNode) != ParentMappingInAbstractionTree.end()) {
      //        pathToTreeRoot.push_back(ParentMappingInAbstractionTree[tmpNode]);
      //        tmpNode = ParentMappingInAbstractionTree[tmpNode];
      //      }
      CodeGen->EmitCode("if (progress[tid_to_run]) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("progress[tid_to_run]->ref = \""+SPSNodeNameTable[currentNode]+"\"");
      CodeGen->EmitStmt("progress[tid_to_run]->iteration = parallel_iteration_vector");
      CodeGen->EmitStmt("progress[tid_to_run]->chunk = c");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      codess << "Progress *p = new Progress(\"" << SPSNodeNameTable[currentNode] << "\", ";
      codess << "parallel_iteration_vector, c)";
      CodeGen->EmitStmt(codess.str());
      codess.str("");
      CodeGen->EmitStmt("progress[tid_to_run] = p");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      if (scheduling == STATIC) {
        CodeGen->EmitStmt(
            "idle_threads_iterator = idle_threads.erase("
            "idle_threads_iterator)");
      } else {
        CodeGen->EmitStmt("idle_threads.erase(idle_threads.begin())");
      }
      CodeGen->EmitStmt("worker_threads.emplace_back(tid_to_run)");
      CodeGen->EmitDebugInfo(
          "cout << \"Move \" << tid_to_run << \" to worker_threads\" << endl");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} /* end of progress assignment */");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} /* end of chunk availability check */");
      /* Generate the Interleaving code for the current outermost loop */
      // TODO: here we use uniform interleaving.
      //  more complex interleaving model should be added and the code should be
      //  controled by the user
      CodeGen->EmitLabel("INTERLEAVING_LOOP_"+to_string(ParallelLoopNode->getLoopID()));
      if (type == UNIFORM_INTERLEAVING) {
        CodeGen->EmitComment("UNIFORM INTERLEAVING");
        CodeGen->EmitStmt("working_threads = (int)worker_threads.size()");
        CodeGen->EmitStmt("sort(worker_threads.begin(), worker_threads.end())");
        CodeGen->EmitStmt(
            "auto worker_thread_iterator = worker_threads.begin()");
        CodeGen->EmitCode(
            "while (worker_thread_iterator != worker_threads.end()) {");
        CodeGen->tab_count++;
        CodeGen->EmitStmt(
            "tid_to_run = *worker_thread_iterator");
#if 0
        CodeGen->EmitCode(
            "for (tid_to_run = 0; tid_to_run < THREAD_NUM; tid_to_run++) {");
        CodeGen->tab_count++;
#endif
      } else {
        CodeGen->EmitComment("RANDOM INTERLEAVING");
        CodeGen->EmitCode("while(!worker_threads.empty()) {");
        CodeGen->tab_count++;
        if (EnableParallelOpt) {
          CodeGen->EmitStmt("uniform_int_distribution<int> distribution(0, worker_threads.size()-1)");
          CodeGen->EmitStmt(
              "tid_to_run = worker_threads[distribution(generator)]");
        } else {
          CodeGen->EmitStmt(
              "tid_to_run = worker_threads[rand()%worker_threads.size()]");
        }
      }
      CodeGen->EmitCode("if (!progress[tid_to_run]->isInBound()) {");
      CodeGen->tab_count++;
      //      errs() << "#ifdef SIMULATOR_DEBUG\n";
      //      errs() << space << space << space << space << "cout << \"[\" << tid_to_run << \"] \" << progress[tid_to_run].iteration[0] << \" > \" << progress[tid_to_run].chunk.second << endl;\n"; errs() << "#endif\n";
      //				errs() << space << space << space << space << "candidate_thread_pool.insert(tid_to_run);\n";
      if (type == UNIFORM_INTERLEAVING) {
        CodeGen->EmitStmt("worker_thread_iterator++");
        CodeGen->EmitDebugInfo("cout << \"[\" << tid_to_run << \"] Out of bound\" << endl;");
      }
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitHashMacro("#if defined(DEBUG)");
      if (EnableSampling && SamplingMethod == RANDOM_START) {
        CodeGen->EmitCode("if (start_reuse_search) {");
        CodeGen->tab_count++;
      }
      CodeGen->EmitStmt("cout << \"[\" << tid_to_run << \"] Access \" << progress[tid_to_run]->ref << \" {\"");
      CodeGen->EmitCode("for (unsigned i = 0; i < progress[tid_to_run]->iteration.size(); i++) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("cout << progress[tid_to_run]->iteration[i]");
      CodeGen->EmitCode("if (i < progress[tid_to_run]->iteration.size()-1) { cout << \",\"; }");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      if (EnableSampling && SamplingMethod == RANDOM_START) {
        CodeGen->EmitStmt("cout << \"} \" << start_reuse_search << endl");
        CodeGen->tab_count--;
        CodeGen->EmitCode("}");
      } else {
        CodeGen->EmitStmt("cout << \"}\" << endl");
      }
      CodeGen->EmitHashMacro("#endif");
//      errs() << "#ifdef SIMULATOR_DEBUG\n";
//      errs() << space << space << space << "cout << \"[\" << tid_to_run << \"] Iterate \" << progress[tid_to_run].ref << \" at \" << progress[tid_to_run].getIteration() << endl;\n";
//      errs() << "#endif\n";
    }
    string currAccName = SPSNodeNameTable[currentNode];//GetSPSTNodeName(currentNode);
    if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(currentNode)) {
      codess << "if (progress[tid_to_run]->ref == \"" << currAccName << "\") {";
      CodeGen->EmitCode(codess.str());
      codess.str("");
      CodeGen->tab_count++;
      // create a condition
      string branch = EmitBranchCondExpr(Branch);
      // CodeGen->EmitComment(to_string(Branch->neighbors[1]->neighbors.size()) + ", FirstAccesses of branch " + branch +" has " + to_string(FirstAccesses.size()) + " accesses");
      if (!Branch->neighbors[0]->neighbors.empty()) {
        SmallVector<SPSTNode *, 8> FirstAccesses;
        G->QueryEngine->FindFirstAccessNodesOfParent(Branch->neighbors[0], FirstAccesses);
        CodeGen->EmitCode(branch + " {");
        CodeGen->tab_count++;
        codess << "progress[tid_to_run]->increment(\"" << SPSNodeNameTable[FirstAccesses[0]] << "\")";
        CodeGen->EmitStmt(codess.str());
        codess.str("");
        if (type == UNIFORM_INTERLEAVING) {
          CodeGen->EmitStmt("worker_thread_iterator++");
        }
        CodeGen->EmitStmt("continue");
        CodeGen->tab_count--;
      }
#if 0
      OpenBranches.push(Branch);
#endif
      if (!Branch->neighbors[1]->neighbors.empty()) {
        SmallVector<SPSTNode *, 8> FirstAccesses;
        G->QueryEngine->FindFirstAccessNodesOfParent(Branch->neighbors[1], FirstAccesses);
        if (Branch->neighbors[0]->neighbors.empty()) {
          CodeGen->EmitCode(branch + " {");
          CodeGen->tab_count++;
          codess << "progress[tid_to_run]->increment(\"" << SPSNodeNameTable[FirstAccesses[0]] << "\")";
          CodeGen->EmitStmt(codess.str());
          codess.str("");
        } else {
          CodeGen->EmitCode("} else {");
          CodeGen->tab_count++;
          codess << "progress[tid_to_run]->increment(\"" << SPSNodeNameTable[FirstAccesses[0]] << "\")";
          CodeGen->EmitStmt(codess.str());
          codess.str("");
        }
        if (type == UNIFORM_INTERLEAVING) {
          CodeGen->EmitStmt("worker_thread_iterator++");
        }
        CodeGen->EmitStmt("continue");
        CodeGen->tab_count--;
      }
      CodeGen->EmitCode("} // end of if condition");
    } else if (RefTNode *RefNode = dynamic_cast<RefTNode *>(currentNode)) {
#if 0
      if (!OpenBranches.empty()) {
        BranchTNode *TopOpenBranch = OpenBranches.top();
        bool FindTrueFirst = currentNode == TopOpenBranch->neighbors[0]->neighbors.front(),
             FindTrueLast = currentNode == TopOpenBranch->neighbors[0]->neighbors.back(),
             FindFalseFirst = (!TopOpenBranch->neighbors[1]->neighbors.empty() &&
                              currentNode == TopOpenBranch->neighbors[1]->neighbors.front()),
             FindFalseLast = (!TopOpenBranch->neighbors[1]->neighbors.empty() &&
                             currentNode == TopOpenBranch->neighbors[1]->neighbors.back());
        // if this is the first ref of the top open if branch
        if (FindTrueFirst) {
          // visit this access
          vector<string> params = EmitRefNodeAccessExpr(RefNode, true);
          if (!params.empty()) {
            CodeGen->EmitFunctionCall("rtTmpAccess", params);
          }
        }

        // if this is the last ref of the top open if branch
        // CodeGen->tab_count--;
        // generate } else { if necessary or } if no false branch
        // CodeGen->tab_count++;
        if (FindTrueLast) {
          // visit this access here
          vector<string> params = EmitRefNodeAccessExpr(RefNode, true);
          if (!params.empty()) {
            CodeGen->EmitFunctionCall("rtTmpAccess", params);
          }
          CodeGen->tab_count--;
          if (!TopOpenBranch->neighbors[1]->neighbors.empty()) {
            CodeGen->EmitCode("} else {");
            CodeGen->tab_count++;
          } else {
            CodeGen->EmitCode("}");
          }
        }

        // if this is the first ref of the top open else branch
        if (FindFalseFirst) {
          // visit this access here
          vector<string> params = EmitRefNodeAccessExpr(RefNode, true);
          if (!params.empty()) {
            CodeGen->EmitFunctionCall("rtTmpAccess", params);
          }
        }

        // if this is the last ref of the top open else branch
        // CodeGen->tab_count--;
        // generate }
        // OpenBranches.pop();
        if (FindFalseLast) {
          // visit this access here
          vector<string> params = EmitRefNodeAccessExpr(RefNode, true);
          if (!params.empty()) {
            CodeGen->EmitFunctionCall("rtTmpAccess", params);
          }
          CodeGen->tab_count--;
          CodeGen->EmitCode("}");
          OpenBranches.pop();
        }

        if (!(FindTrueFirst || FindTrueLast || FindFalseFirst || FindFalseLast)) {
          // if this is the normal reference
          // visit this access here
          vector<string> params = EmitRefNodeAccessExpr(RefNode, true);
          if (!params.empty()) {
            CodeGen->EmitFunctionCall("rtTmpAccess", params);
          }
        }
      } else {h
#endif
      // if this is the normal reference
      codess << "if (progress[tid_to_run]->ref == \"" << currAccName << "\") {";
      CodeGen->EmitCode(codess.str());
      codess.str("");
      CodeGen->tab_count++;
      if (RefNode == TargetRef) {
        CodeGen->EmitCode("if (!start_reuse_search) { start_reuse_search= (start_tid == tid_to_run); }");
      }
      CodeGen->EmitCode("if (start_reuse_search) {");
      CodeGen->tab_count++;
//      if (EnableParallelOpt)
//        CodeGen->EmitStmt("m.lock()");
      if (QueryEngine->areAccessToSameArray(RefNode, TargetRef)) {
        // visit this access here if the RefNode access the
        // same array as the TargetRef
        vector<string> params = EmitRefNodeAccessExpr(RefNode, true);
        if (!params.empty()) {
          AccPath path;
          QueryEngine->GetPath(RefNode, nullptr, path);
          if (path.empty() || SPSNodeNameTable[RefNode] != SPSNodeNameTable[TargetRef] ) {
          } else {
            codess << "Iteration *access = new Iteration(\""
                   << SPSNodeNameTable[RefNode] << "\", {";
            auto pathIter = path.rbegin();
            TranslateStatus status = SUCCESS;
            while (pathIter != path.rend()) {
              LoopTNode *NestLoop = dynamic_cast<LoopTNode *>(*pathIter);
              assert(NestLoop &&
                         "All node in a path should be an instance of LoopTNode");
              string nest_loop_induction = ReplaceInductionVarExprByType(
                  NestLoop->getInductionPhi(), OMP_PARALLEL, status);
              codess << nest_loop_induction;
              if (pathIter + 1 != path.rend())
                codess << ",";
              pathIter++;
            }
            codess << "})";
            CodeGen->EmitStmt(codess.str());
            codess.str("");
          }
//          if (EnableParallelOpt) {
//            params.push_back("this_thread::get_id()");
//            CodeGen->EmitFunctionCall("pluss_sample_access_parallel", params);
//          } else {
//            CodeGen->EmitFunctionCall("pluss_sample_access", params);
//          }
          CodeGen->EmitStmt("string array = " + params[0]);
//          CodeGen->EmitStmt("subscripts = " + params[1]);
          codess << "addr = GetAddress_" << SPSNodeNameTable[RefNode] << "(";
          for (unsigned i = 0; i < RefNode->getSubscripts().size(); i++) {
//            codess << "subscripts[" << i << "]";
            codess << EmitRefNodeAccessExprAtIdx(RefNode->getSubscripts()[i], true);
            if (i < RefNode->getSubscripts().size()-1)
              codess << ",";
          }
          codess << ")";
          CodeGen->EmitStmt(codess.str());
          codess.str("");
          if (EnableSampling && SamplingMethod == RANDOM_START) {
            CodeGen->EmitStmt("bool isSample = false");
            if (RefNode == TargetRef) {
              CodeGen->EmitDebugInfo("cout << access->toString() << \" @ \" << addr << endl");
              CodeGen->EmitCode(
                  "if (access->compare(start_iteration) == 0) {");
              CodeGen->tab_count++;
              CodeGen->EmitDebugInfo("cout << \"Meet the start sample \" << access->toString() << endl");
              CodeGen->EmitStmt("isSample = true");
              CodeGen->tab_count--;
              CodeGen->EmitCode("} else if ((!samples.empty() && access->compare(samples.top()) == 0) "
                                "|| (sample_names.find(access->toAddrString()) != sample_names.end())) {");
              CodeGen->tab_count++;
              CodeGen->EmitDebugInfo(
                  "cout << \"Meet a new sample \" << access->toString() << \" while searching reuses\" << endl");
              CodeGen->EmitStmt("traversing_samples++");
              CodeGen->EmitCode("if (!samples.empty() && access->compare(samples.top()) == 0) {");
              CodeGen->tab_count++;
              CodeGen->EmitStmt("samples.pop()");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->EmitStmt("isSample = true");
              CodeGen->EmitStmt("samples_meet.insert(access->toAddrString())");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
            }
          }

          CodeGen->EmitCode("if (LAT[tid_to_run].find(addr) != LAT[tid_to_run].end()) {");
          CodeGen->tab_count++;
          CodeGen->EmitStmt("long reuse = count[tid_to_run] - LAT[tid_to_run][addr]");

          // Iteration *src and *access forms a reuse
          // Find RefTNode of src and sink (TargetRef and RefNode)
          bool src_has_parallel_induction = false, sink_has_parallel_induction = false;
          for (auto Index : TargetRef->getSubscripts()) {
            src_has_parallel_induction |= hasParallelLoopInductionVar(Index);
          }
          for (auto Index : RefNode->getSubscripts()) {
            sink_has_parallel_induction |= hasParallelLoopInductionVar(Index);
          }
          CodeGen->EmitComment("Src has Parallel Induction : " + to_string(src_has_parallel_induction));
          CodeGen->EmitComment("Sink has Parallel Induction : " + to_string(sink_has_parallel_induction));

          if (InterleavingTechnique == 1) {
            // compute the expression of one iteration of parallel loop
            // replace all induction variable names inside the expression with
            // the progress->iteration vector.
            LoopTNode *ParallelLoopDominator = QueryEngine->GetParallelLoopDominator(RefNode);
            string parallel_iteration_cnt = EmitOneParallelIterationTripFormula(ParallelLoopDominator);
            AccPath PathToParallelLoop;
            QueryEngine->GetPath(RefNode, ParallelLoopDominator, PathToParallelLoop);
            auto pathIter = PathToParallelLoop.rbegin();
            int distance = isInsideParallelLoopNest(ParallelLoopDominator->getInductionPhi());
            string to_replace = "progress[tid_to_run]->iteration[" + to_string(distance) + "]";
            ReplaceSubstringWith(parallel_iteration_cnt,
                                 InductionVarNameTable[ParallelLoopDominator->getInductionPhi()],
                                 to_replace);
            while (pathIter != PathToParallelLoop.rend()) {
              LoopTNode *NestLoop = dynamic_cast<LoopTNode *>(*pathIter);
              assert(NestLoop && "All node in a path should be an instance of LoopTNode");
              distance = isInsideParallelLoopNest(NestLoop->getInductionPhi());
              to_replace = "progress[tid_to_run]->iteration[" + to_string(distance) + "]";
              ReplaceSubstringWith(parallel_iteration_cnt,
                                   InductionVarNameTable[NestLoop->getInductionPhi()],
                                   to_replace);
              pathIter++;
            }
            switch (isInterThreadSharingRef(RefNode)) {
            case FULL_SHARE: {
              // sharing THREAD_NUM -1 elements
              CodeGen->EmitCode(
                  "if (distance_to(reuse,0) > distance_to(reuse," +
                  parallel_iteration_cnt + ")) {");
              CodeGen->tab_count++;
              if (EnableParallelOpt) {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"share_histogram[THREAD_NUM-1]", "reuse", "1."});
              } else {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"share_intra_thread_RI[THREAD_NUM-1]", "reuse", "1."});
              }
              CodeGen->tab_count--;
              CodeGen->EmitCode("} else {");
              CodeGen->tab_count++;
//              if (EnableParallelOpt) {
//                CodeGen->EmitFunctionCall(
//                    "pluss_parallel_histogram_update",
//                    {"share_histogram[THREAD_NUM-1]", "reuse", "1."});
//              } else {
//                CodeGen->EmitFunctionCall(
//                    "pluss_parallel_histogram_update",
//                    {"share_intra_thread_RI[THREAD_NUM-1]", "reuse", "1."});
//              }
              if (EnableParallelOpt) {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_histogram", "reuse", "1."});
              } else {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_intra_thread_RI", "reuse", "1."});
              }
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              break;
            }
            case SPATIAL_SHARE: {
              // sharing CLS/(DS*C)
              CodeGen->EmitCode("if (CHUNK_SIZE < (CLS / DS)) {");
              CodeGen->tab_count++;
              CodeGen->EmitCode(
                  "if (distance_to(reuse,0) > distance_to(reuse," +
                  parallel_iteration_cnt + ")) {");
              CodeGen->tab_count++;
              if (EnableParallelOpt) {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"share_histogram[CLS/(DS*CHUNK_SIZE)-1]", "reuse",
                     "1"});
              } else {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"share_intra_thread_RI[CLS/(DS*CHUNK_SIZE)-1]", "reuse",
                     "1"});
              }
              CodeGen->tab_count--;
              CodeGen->EmitCode("} else {");
              CodeGen->tab_count++;
              if (EnableParallelOpt) {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_histogram", "reuse", "1"});
              } else {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_intra_thread_RI", "reuse", "1."});
              }
//              if (EnableParallelOpt) {
//                CodeGen->EmitFunctionCall(
//                    "pluss_parallel_histogram_update",
//                    {"share_histogram[CLS/(DS*CHUNK_SIZE)-1]", "reuse",
//                     "1"});
//              } else {
//                CodeGen->EmitFunctionCall(
//                    "pluss_parallel_histogram_update",
//                    {"share_intra_thread_RI[CLS/(DS*CHUNK_SIZE)-1]", "reuse",
//                     "1"});
//              }
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->tab_count--;
              CodeGen->EmitCode("} else {");
              CodeGen->tab_count++;
              if (EnableParallelOpt) {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_histogram", "reuse", "1."});
              } else {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_intra_thread_RI", "reuse", "1."});
              }
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              break;
            }
            default:
              if (EnableParallelOpt) {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_histogram", "reuse", "1."});
              } else {
                CodeGen->EmitFunctionCall(
                    "pluss_parallel_histogram_update",
                    {"no_share_intra_thread_RI", "reuse", "1."});
              }
              break;
            }
          }



//          if (EnableParallelOpt)
//            CodeGen->EmitStmt("pluss_parallel_histogram_update(histogram, reuse, 1)");
//          else
//            CodeGen->EmitStmt("pluss_histogram_update(reuse, 1)");

          CodeGen->EmitStmt("Iteration *src = LATSampleIterMap[tid_to_run][LAT[tid_to_run][addr]]");
          if (EnableSampling && SamplingMethod == RANDOM_START) {
            if (RefNode == TargetRef) {
              CodeGen->EmitDebugInfo(
                  "cout << \"[\" << reuse << \"] \" << src->toString() << \" -> \" << access->toString() << endl");
            }
            CodeGen->EmitStmt("traversing_samples--");
            CodeGen->EmitComment("stop traversing if reuse of all samples"
                                 "are found");
            CodeGen->EmitDebugInfo(
                "if (samples.empty() && traversing_samples == 0) { cout << \"[\" << reuse << \"] for last sample \" << src->toString() << endl; }");
            CodeGen->EmitCode("if (samples.empty() && traversing_samples == 0) { goto END_SAMPLE; }");
            CodeGen->EmitCode("if (traversing_samples == 0) { ");
            CodeGen->tab_count++;
            CodeGen->EmitDebugInfo(
                "cout << \"delete sample \" << src->toString() << \", active:\" << traversing_samples << \", remain:\" << samples.size() << endl");
            CodeGen->EmitStmt("delete src");

            CodeGen->EmitStmt("LATSampleIterMap[tid_to_run].erase(LAT[tid_to_run][addr])");
            CodeGen->EmitStmt("LAT[tid_to_run].erase(addr)");

            CodeGen->EmitComment("Here we examine if there is an out-of-order effect.");
            CodeGen->EmitComment("if the next sample we should jump has been traversed before, we will pop this sample directly.");
            CodeGen->EmitComment("It is safe to call samples.top() once, since when entering here, 'samples' queue is not empty()");
            CodeGen->EmitCode("if (samples_meet.size() >= samples.size()) { goto END_SAMPLE; }");
            CodeGen->EmitStmt("Iteration next = samples.top()");
            CodeGen->EmitCode("while(samples_meet.find(next.toAddrString()) != samples_meet.end()) {");
            CodeGen->tab_count++;
            CodeGen->EmitDebugInfo("cout << \"Skip \" << next.toString() << \" because we met this sample already \" << endl");
            CodeGen->EmitStmt("samples.pop()");
            CodeGen->EmitComment("All samples has been traversed, no need to jump");
            CodeGen->EmitCode("if (samples.empty()) { break; }");
            CodeGen->EmitStmt("next = samples.top()");
            CodeGen->tab_count--;
            CodeGen->EmitCode("} // end of out-of-order check");
            CodeGen->EmitCode("if (!samples.empty()) {");
            CodeGen->tab_count++;
            CodeGen->EmitHashMacro("#if defined(DEBUG)");
            CodeGen->EmitStmt("next = samples.top()");
            CodeGen->EmitStmt("cout << \"Jump to next sample \" << next.toString() << endl");
            CodeGen->EmitHashMacro("#endif");
            CodeGen->EmitStmt("goto START_SAMPLE_"+SPSNodeNameTable[TargetRef]);
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else { goto END_SAMPLE; }");
            CodeGen->tab_count--;
            CodeGen->EmitCode("} // end of if (traversing_samples == 0)");
          }
          CodeGen->EmitDebugInfo("cout << \"delete sample \" << src->toString() << \", active:\" << traversing_samples << \", remain:\" << samples.size() << endl");
          CodeGen->EmitStmt("delete src");

          CodeGen->EmitStmt("LATSampleIterMap[tid_to_run].erase(LAT[tid_to_run][addr])");
          CodeGen->EmitStmt("LAT[tid_to_run].erase(addr)");
          CodeGen->tab_count--;
          CodeGen->EmitCode("} // end of if (LAT.find(addr) != LAT.end())");
          if (RefNode == TargetRef) {
            if (EnableSampling && SamplingMethod == RANDOM_START) {
#if 0
              CodeGen->EmitCode(
                  "if (samples.find(*access) != samples.end()) {");
#endif
              CodeGen->EmitCode("if (isSample) {");
              CodeGen->tab_count++;

              CodeGen->EmitStmt("LAT[tid_to_run][addr] = count[tid_to_run]");
              CodeGen->EmitStmt("LATSampleIterMap[tid_to_run][count[tid_to_run]] = access");

              CodeGen->tab_count--;
              CodeGen->EmitCode("} else { delete access; }");
            } else {
              CodeGen->EmitStmt("LAT[addr] = count");
            }
          }
        }

        CodeGen->EmitStmt("count[tid_to_run]++");
      } else {
        // bypass this accses otherwise
        CodeGen->EmitComment(RefNode->getRefExprString()
                             + " will not access " + TargetRef->getArrayNameString());
        CodeGen->EmitStmt("count[tid_to_run] += 1");

      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of if (start_reuse_search)");
    } // end of node type check
    GraphNodeIter++;
    if (GraphNodeIter == NodesInGraph.end()) {
      // this is the last node in the access graph
      // the next access node to exec is the header node
      GraphNodeIter = NodesInGraph.begin();
      reachTheLastAccessNode = true;
    }
    // increment the progress obj to move to the next node (refnode or branch)
    nextNode = (*GraphNodeIter);
    string nextAccName = SPSNodeNameTable[nextNode];// GetSPSTNodeName(nextNode);
    // check whether we need to insert the increment funciton
    //
    // currentNode stores the current access to be examined, nextNode stores
    // the next access to be examined.
    //
    // If this access is the last access of a loop, then we need to generate the
    // loop increment function. In this case, we will query all edges that sourced
    // by the currentNode.
    //
    // Then there are three cases:
    // CASE 1:
    // Edge (currentNode -> XXX) is carried by a loop (backedge) and this loop is not
    // the immediate loop dominator of currentNode
    //
    // We need to compute loops between currentNode and the carried loop, L1;
    // loops between XXX and the carried loop, L2.
    // Then the iteration vector will pop L1.size times, and push the LB of all
    // loops in L2.
    // We also need to generate the loop IV increment code for IV of carried loop
    //
    // CASE 2:
    // Edge (currentNode -> nextNode)
    //
    // This has two cases:
    // a) currentNode is the last access node
    // b) others
    //
    // For a): We need to firstly compute the immediate dominator that dominates both
    // currentNode and nextNode. Then we do the same as a): compute loops between
    // currentNode and the common immediate dominator, L1; compute loops between nextNode
    // and the common immediate dominator, L2. Pop the iteration vector L1.size times,
    // push the LB of all loops in L2.LA
    //
    // For b): The same as a) but there is one more step, increment the LIV of the outermost
    // loop.
    //
    // CASE 3:
    // Edge (currentNode -> XXX) is carried by a loop (backedge) and this loop is the
    // immediate loop dominator of currentNode.
    //
    // increment the LIV of the carried loop
    //
    if (G->QueryEngine->isLastAccessInLoop(currentNode) ||
        !G->QueryEngine->areTwoAccessInSameLevel(currentNode, nextNode)) {
      if (G->QueryEngine->isLastAccessInLoop(currentNode)) {
        commentss << currAccName << " is the last access node in a loop";
      } else if (!G->QueryEngine->areTwoAccessInSameLevel(currentNode, nextNode)) {
        commentss << currAccName << " and " << nextAccName << " are not in the same loop level";
      }
      SmallVector<AccGraphEdge *, 8> edges;
      G->GetEdgesWithSource(currentNode, edges);
      SPSTNode *sink = nullptr;
      LoopTNode *carryLoop = nullptr;
      for (auto edge : edges) { // iterate the edges from inner to outer
        carryLoop = edge->getCarryLoop();
        sink = edge->getSink();
        nextAccName = SPSNodeNameTable[sink];
        if (carryLoop) { // CASE 1 or CASE 3
          // CASE 3
          if (G->QueryEngine->GetImmdiateLoopDominator(currentNode) == carryLoop) {
            // increment the LIV only
            CodeGen->EmitComment("CASE 3");
            LoopIterUpdateGen(G, edge, carryLoop, true,
                              !G->QueryEngine->areTwoAccessInSameLevel(currentNode, sink));
            codess << "progress[tid_to_run]->increment(\"" << nextAccName << "\")";
            CodeGen->EmitStmt(codess.str());
            codess.str("");
            if (type == UNIFORM_INTERLEAVING) {
              CodeGen->EmitStmt("worker_thread_iterator++");
            }
            CodeGen->EmitStmt("continue");
            CodeGen->tab_count--;
            codess << "} /* end of check to " << currAccName << " */";
            CodeGen->EmitCode(codess.str());
            codess.str("");
          } else { // CASE 1
            CodeGen->EmitComment("CASE 1");
            // pop iteration vector L1.size() times
            // push LB of loop in L2 in iteration vector
            // and increment the LIV
            LoopIterUpdateGen(G, edge, carryLoop, true, true);
            codess << "progress[tid_to_run]->increment(\"" << nextAccName << "\")";
            CodeGen->EmitStmt(codess.str());
            codess.str("");
            if (type == UNIFORM_INTERLEAVING) {
              CodeGen->EmitStmt("worker_thread_iterator++");
            }
            CodeGen->EmitStmt("continue");
            CodeGen->tab_count--;
            codess << "} /* end of check to " << currAccName << " */";
            CodeGen->EmitCode(codess.str());
            codess.str("");
          }
        } else if (edge->hasSource(currentNode) && edge->hasSink(nextNode)) { // CASE 2
          CodeGen->EmitComment("CASE 2");
          SmallVector<SPSTNode *, 8> dominator;
          G->QueryEngine->GetImmediateCommonLoopDominator(currentNode, sink,
                                                          reachTheLastAccessNode,
                                                          dominator);
          // all access nodes are dominated by the parallel loop
          // dominators.size() should >= 1
          assert(dominator.size() == 1);
          assert(dynamic_cast<LoopTNode *>(dominator.front()) && "All dominatos"
                                                                 "should be "
                                                                 "LoopTNode type");
          LoopTNode *ImmCommonDom = dynamic_cast<LoopTNode *>(dominator.front());
          // pop iteration vector L1.size() times
          // push LB of loop in L2 in iteration vector
          // increment the LIV if and only currentNode is the last access node
          LoopIterUpdateGen(G, edge, ImmCommonDom, reachTheLastAccessNode,
                            true);
          codess << "progress[tid_to_run]->increment(\"" << nextAccName << "\")";
          CodeGen->EmitStmt(codess.str());
          codess.str("");
          if (type == UNIFORM_INTERLEAVING) {
            CodeGen->EmitStmt("worker_thread_iterator++");
          }
          CodeGen->EmitStmt("continue");
          CodeGen->tab_count--;
          codess << "} /* end of check to " << currAccName << " */";
          CodeGen->EmitCode(codess.str());
          codess.str("");
        }
      }
    } else {
      stringstream codess;
      codess << "progress[tid_to_run]->increment(\"" << nextAccName << "\")";
      CodeGen->EmitStmt(codess.str());
      codess.str("");
      if (type == UNIFORM_INTERLEAVING) {
        CodeGen->EmitStmt("worker_thread_iterator++");
      }
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      codess << "} /* end of check to " << currAccName << " */";
      CodeGen->EmitCode(codess.str());
      codess.str("");
      if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(currentNode)) {
        if (!Branch->neighbors[0]->neighbors.empty())
          PerRefParallelSamplerBranchBodyGenImpl(dynamic_cast<DummyTNode *>(Branch->neighbors[0]),
                                                 type, nextNode, TargetRef);
        if (!Branch->neighbors[1]->neighbors.empty())
          PerRefParallelSamplerBranchBodyGenImpl(dynamic_cast<DummyTNode *>(Branch->neighbors[1]),
                                                 type, nextNode, TargetRef);
      }
    }

    if (reachTheLastAccessNode)
      break;
  } // end of the while(true) loop

  // handle the interleaving part
  codess.str("");
  CodeGen->EmitCode("if (find(idle_threads.begin(), "
                    "idle_threads.end(), tid_to_run) == "
                    "idle_threads.end()) {");
  CodeGen->tab_count++;
  CodeGen->EmitDebugInfo("cout << \"Move \" << tid_to_run << \" to idle threads\" << endl");
  CodeGen->EmitStmt("idle_threads.emplace_back(tid_to_run)");
  // remove the tid_to_run from worker_threads.
  if (type == UNIFORM_INTERLEAVING) {
    CodeGen->EmitStmt("worker_thread_iterator = worker_threads.erase(find(worker_threads.begin(), worker_threads.end(), tid_to_run))");
  } else if (type == RANDOM_INTERLEAVING) {
    CodeGen->EmitStmt("worker_threads.erase(find(worker_threads.begin(), worker_threads.end(), tid_to_run))");
    CodeGen->EmitStmt("goto CHUNK_ASSIGNMENT_LOOP_" +
                      to_string(ParallelLoopNode->getLoopID()));
  }
#ifdef RANDOM_INTERLEAVING
  errs() << space << space << space << space << space << "threads_to_exec.erase(find(threads_to_exec.begin(), threads_to_exec.end(), tid_to_run));\n";
		errs() << space << space << space << space << space << "goto chunk_assignment_loop_" << to_string(LoopID) << ";\n";
#endif
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} /* end of thread interleaving loop */");
  CodeGen->EmitCode("if (idle_threads.size() == THREAD_NUM && !dispatcher.hasNextChunk("
                    +to_string(scheduling==STATIC)+")) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("break");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} /* end of break condition check */");
#ifdef ENABLE_RANDOM_SAMPLE
  errs() << "end_sample_" << LoopID << ":\n";
  errs() << space << "break;\n";
#endif
  CodeGen->tab_count--;
  CodeGen->EmitCode("} /* end of while(true) */");
  CodeGen->EmitComment("reset both lists so they can be reused for later parallel loop");
  CodeGen->EmitStmt("idle_threads.clear()");
  CodeGen->EmitStmt("worker_threads.clear()");

  CodeGen->EmitCode("for (unsigned i = 0; i < progress.size(); i++) {");
  CodeGen->tab_count++;
  CodeGen->EmitCode("if (progress[i]) {");
  CodeGen->tab_count++;
  CodeGen->EmitStmt("delete progress[i]");
  CodeGen->EmitStmt("progress[i] = nullptr");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  CodeGen->tab_count--;
  CodeGen->EmitCode("} // end of progress traversal");
}

void ModelCodeGenProWrapperPass::ParallelSamplerSamplingHeaderGen(LoopTNode *ParallelLoop,
                                                                  AccGraph *G,
                                                                  RefTNode *TargetRef,
                                                                  Interleaving type,
                                                                  SchedulingType schedule)
{
  AccPath pathToTreeRoot, pathToParallelTreeRoot;
  QueryEngine->GetPath(ParallelLoop, nullptr, pathToTreeRoot);
  G->QueryEngine->GetPath(TargetRef, nullptr, pathToParallelTreeRoot);
  unsigned DistanceToTop = 0, DistanceToParallel = 0; // 0 means the top loop is parallelized
  if (!pathToTreeRoot.empty()) { DistanceToTop = pathToTreeRoot.size(); }
  if (!pathToParallelTreeRoot.empty()) { DistanceToParallel = pathToParallelTreeRoot.size(); }
  stringstream ss;
  TranslateStatus status = SUCCESS;
  CodeGen->EmitStmt("dispatcher.setStartPoint(start_iteration.ivs[" +
                    to_string(DistanceToTop) + "])");
  // we can assume the chunk dispatcher had already been created
  // this method will be called only in the RandomSampling approach
  // is enabled (EnableSampling && SamplingMethod == RANDOM_START)
  if (type == UNIFORM_INTERLEAVING || type == RANDOM_INTERLEAVING) {
    // in uniform interleaving, all threads start at the same place,
    // we need to compute the value of the parallel loop iteration
    // and copy the rest iteration variable value
    if (schedule == STATIC) {
      CodeGen->EmitStmt("start_tid = dispatcher.getStaticTid(start_iteration.ivs["
                        + to_string(DistanceToTop)+"])");
      CodeGen->EmitCode("for (auto idle_threads_iterator = idle_threads.begin();"
                        "idle_threads_iterator != idle_threads.end();) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("int idle_tid = *idle_threads_iterator");
      CodeGen->EmitCode("if (dispatcher.hasNextStaticChunk(idle_tid)) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("Chunk start_chunk = "
                        "dispatcher.getStaticStartChunk("
                        "start_iteration.ivs[" +
                        to_string(DistanceToTop) + "], idle_tid)");
      CodeGen->EmitCode("if (start_chunk.first > start_chunk.second) { ");
      CodeGen->tab_count++;
      CodeGen->EmitComment("though the idle_tid has an available chunk, but at the given "
                           "sampling point, there is no iteration to execute for this idle_tid");
      CodeGen->EmitDebugInfo("cout << \"[\" << start_chunk.first << \">\" << start_chunk.second "
                             "<< \"] No available iteration for \" << idle_tid <<  endl");
      CodeGen->EmitStmt("idle_threads_iterator++");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitCode("if (progress[idle_tid]) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("progress[idle_tid]->ref = \""+SPSNodeNameTable[TargetRef]+"\"");
      ss << "{";
      auto path_riter = pathToParallelTreeRoot.rbegin();
      DistanceToParallel = 0;
      while (path_riter != pathToParallelTreeRoot.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode type");
        if (LoopInPath == ParallelLoop) { // this is the parallel loop
          ss << "start_chunk.first";
        } else if (LoopInPath->getLoopLevel() <
                   ParallelLoop->getLoopLevel()) { // this is the parent of the parallel loop
          ss << Translator->ValueToStringExpr(LoopInPath->getInductionPhi(),
                                              status);
        } else { // this is the loop inside the parallel loop
          ss << "start_iteration.ivs[" << (DistanceToParallel+DistanceToTop) << "]";
        }
        if (path_riter != pathToTreeRoot.rend() - 1)
          ss << ", ";
        path_riter++;
        DistanceToParallel++;
      }
      ss << "}";
      CodeGen->EmitStmt("progress[idle_tid]->iteration = "+ss.str());
      CodeGen->EmitStmt("progress[idle_tid]->chunk = start_chunk");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      ss << ", start_chunk)";
      CodeGen->EmitStmt("Progress *p = new Progress(\""+SPSNodeNameTable[TargetRef]+"\", "+ss.str());
      ss.str("");
#if 0
      CodeGen->EmitStmt("per_thread_start_point[idle_tid] = p");
#endif
      CodeGen->EmitStmt("progress[idle_tid] = p");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitDebugInfo("cout << \"[\" << idle_tid << \"] starts from \" << progress[idle_tid]->toString() << endl");
      CodeGen->EmitStmt("idle_threads_iterator = idle_threads.erase(idle_threads_iterator)");
      CodeGen->EmitDebugInfo("cout << \"Move \" << idle_tid << \" to worker_threads\" << endl");
      CodeGen->EmitStmt("worker_threads.push_back(idle_tid)");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitStmt("idle_threads_iterator++");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
    } else if (schedule == DYNAMIC) {
      // in dynamic scheduling, there is no static relatioship between an iteration
      // and thread
      // we can randomly select a tid, then the start_iteration.ivs belongs to this
      // tid.
      string loop_step = "1";
      loop_step = Translator->ValueToStringExpr(
          ParallelLoop->getLoopBound()->StepValue, status);
      CodeGen->EmitStmt("random_shuffle(idle_threads.begin(), "
                        "idle_threads.end(), [](int n) { return rand() % n; })");
      CodeGen->EmitStmt("int next_chunk_ptr = 0, prev_chunk_ptr = 0, offset = 0");
      CodeGen->EmitStmt("vector<Chunk> next_chunks, prev_chunks");
      CodeGen->EmitStmt("Chunk start_chunk = dispatcher.getStartChunk(start_iteration.ivs["
                        +to_string(DistanceToTop)+"])");
      CodeGen->EmitComment("we get THREAD_NUM-1 next chunks and "
                           "THREAD_NUM-1 previous chunks");
      CodeGen->EmitStmt("dispatcher.getPrevKChunksFrom(THREAD_NUM-1, "
                        "start_chunk, prev_chunks)");
      CodeGen->EmitStmt("dispatcher.getNextKChunksFrom(THREAD_NUM-1, "
                        "start_chunk, next_chunks)");
      CodeGen->EmitStmt("auto iter = idle_threads.begin()");
      CodeGen->EmitStmt("int first_idle_tid = *iter");
      CodeGen->EmitCode("while (iter != idle_threads.end()) {");
      CodeGen->tab_count++;
      // in the dynamic scheduling, we assign the start_iteration.ivs as the
      // start point of the first tid in the random_shuffle of idle_threads
      // for the rest, if tid < start_tid, we assign it prev_chunks, others
      // assign next_chunks
      CodeGen->EmitCode("if (*iter == first_idle_tid) {");
      CodeGen->tab_count++;
      CodeGen->EmitCode("if (progress[*iter]) {");
      CodeGen->tab_count++;
      ss << "{";
      auto path_riter = pathToParallelTreeRoot.rbegin();
      int i = 0;
      while (path_riter != pathToParallelTreeRoot.rend()) {
        ss << "start_iteration.ivs[" << DistanceToTop+i << "]";
        if (path_riter < pathToParallelTreeRoot.rend()-1)
          ss << ",";
        path_riter++;
        i++;
      }
      ss << "}";
      CodeGen->EmitStmt("progress[*iter]->ref = \""+SPSNodeNameTable[TargetRef]+"\"");
      CodeGen->EmitStmt("progress[*iter]->iteration = " + ss.str());
      CodeGen->EmitStmt("progress[*iter]->chunk = start_chunk");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      ss << ",start_chunk)";
      CodeGen->EmitStmt("Progress *p = new Progress(\""+SPSNodeNameTable[TargetRef]+"\","+ss.str());
      ss.str("");
#if 0
      CodeGen->EmitStmt("per_thread_start_point[*iter] = p");
#endif
      CodeGen->EmitStmt("progress[*iter] = p");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitDebugInfo("cout << \"[\" << *iter << \"] starts from \" << progress[*iter]->toString() << endl");
      if (dyn_cast<ConstantInt>(ParallelLoop->getLoopBound()->StepValue)->isNegative()) {
        CodeGen->EmitStmt("offset = (start_chunk.second - start_iteration.ivs[" +
                          to_string(DistanceToTop) + "])/"+loop_step);
      } else {
        CodeGen->EmitStmt("offset = (start_iteration.ivs[" +
                          to_string(DistanceToTop) + "]-start_chunk.first)/"+loop_step);
      }
      CodeGen->EmitDebugInfo("cout << \"Move \" << *iter << \" to worker_threads\" << endl");
      CodeGen->EmitStmt("worker_threads.push_back(*iter)");
      CodeGen->EmitStmt("iter = idle_threads.erase(iter)");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("int parallel_loop_start = -1");
      CodeGen->EmitCode("if (*iter > first_idle_tid "
                        "|| prev_chunk_ptr == prev_chunks.size()) {");
      CodeGen->tab_count++;
      CodeGen->EmitCode("if (next_chunk_ptr < next_chunks.size()) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("start_chunk = next_chunks[next_chunk_ptr]");
      CodeGen->EmitStmt("next_chunk_ptr++");
      // here we need to update the bound of the next available chunk to be
      // assigned, since its current "next chunk" has already been chosen as the
      // start point of another thread
      CodeGen->EmitStmt("dispatcher.moveToNextChunk()");
//      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
//                        "dist_chunk(start_chunk.first, start_chunk.second)");
//      CodeGen->EmitStmt("parallel_loop_start = dist_chunk(rng)");
      if (dyn_cast<ConstantInt>(ParallelLoop->getLoopBound()->StepValue)->isNegative()) {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second + (offset*("+loop_step+"))");
      } else {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first + (offset*"+loop_step+")");
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else if (*iter < first_idle_tid) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("start_chunk = prev_chunks[prev_chunk_ptr]");
      CodeGen->EmitStmt("prev_chunk_ptr++");
//      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
//                        "dist_chunk(start_chunk.first, start_chunk.second)");
//      CodeGen->EmitStmt("parallel_loop_start = dist_chunk(rng)");
      if (dyn_cast<ConstantInt>(ParallelLoop->getLoopBound()->StepValue)->isNegative()) {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second + (offset*("+loop_step+"))");
      } else {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first + (offset*"+loop_step+")");
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      if (dyn_cast<ConstantInt>(ParallelLoop->getLoopBound()->StepValue)->isNegative()) {
        CodeGen->EmitCode("if (parallel_loop_start > 0 && parallel_loop_start >= start_chunk.first) {");
      } else {
        CodeGen->EmitCode("if (parallel_loop_start > 0 && parallel_loop_start <= start_chunk.second) {");
      }
      CodeGen->tab_count++;
      // Generate code that sample the rest of the iteration
      CodeGen->EmitCode("if (progress[*iter]) {");
      CodeGen->tab_count++;
      ss << "{";
      path_riter = pathToParallelTreeRoot.rbegin();
      i = 0;
      while (path_riter != pathToParallelTreeRoot.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode type");
        if (LoopInPath == ParallelLoop) { // this is the parallel loop
          ss << "parallel_loop_start";
        } else if (LoopInPath->getLoopLevel() <
                   ParallelLoop->getLoopLevel()) { // this is the parent of the parallel loop
          ss << Translator->ValueToStringExpr(LoopInPath->getInductionPhi(),
                                              status);
        } else { // this is the loop inside the parallel loop
//          ss << ReplaceInductionVarExprByType(
//              LoopInPath->getLoopBound()->InitValue, SAMPLE, status);
          ss << "start_iteration.ivs[" << (i+DistanceToTop) << "]";
        }
        if (path_riter != pathToParallelTreeRoot.rend() - 1)
          ss << ", ";
        path_riter++;
        i++;
      }
      ss << "}";
      CodeGen->EmitStmt("progress[*iter]->ref = \""+SPSNodeNameTable[TargetRef]+"\"");
      CodeGen->EmitStmt("progress[*iter]->iteration = "+ss.str());
      CodeGen->EmitStmt("progress[*iter]->chunk = start_chunk");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      ss << ", start_chunk)";
      CodeGen->EmitStmt("Progress *p = new Progress(\""+SPSNodeNameTable[TargetRef]+"\","+ss.str());
      ss.str("");
#if 0
      CodeGen->EmitStmt("per_thread_start_point[*iter] = p");
#endif
      CodeGen->EmitStmt("progress[*iter] = p");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitDebugInfo("cout << \"[\" << *iter << \"] starts from \" << progress[*iter]->toString() << endl");
      CodeGen->EmitDebugInfo("cout << \"Move \" << *iter << \" to worker_threads\" << endl");
      CodeGen->EmitStmt("worker_threads.push_back(*iter)");
      CodeGen->EmitStmt("iter = idle_threads.erase(iter)");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of if (parallel_loop_start > 0)");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // if (iter != idle_threads.begin())");
      CodeGen->EmitStmt("iter++");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of idle_threads traversal");
    }
  } else {
    // in random interleaving, threads can start at different place,
    // we need to compute the value of the parallel loop iteration
    // and copy the rest iteration variable value
    CodeGen->EmitStmt("int parallel_loop_start = -1");
    if (schedule == STATIC) {
      // the following code will be generated by the SamplerCodeGen
      CodeGen->EmitCode("for (auto idle_threads_iterator = idle_threads.begin(); idle_threads_iterator != idle_threads.end();) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("int idle_tid = *idle_threads_iterator");
      CodeGen->EmitCode("if (idle_tid != dispatcher.getStaticTid(start_iteration.ivs["
                        +to_string(DistanceToTop)+"])) {");
      CodeGen->tab_count++;
//      CodeGen->EmitStmt("int start_cid = dispatcher.getStaticChunkID(start_iteration.ivs["
//                        +to_string(DistanceToTop)+"])");
      CodeGen->EmitCode("if (dispatcher.hasNextStaticChunk(idle_tid)) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("Chunk start_chunk = dispatcher.getNextStaticChunk(idle_tid)");
      CodeGen->EmitCode("if (start_chunk.first > start_chunk.second) { ");
      CodeGen->tab_count++;
      CodeGen->EmitComment("though the idle_tid has an available chunk, but at the given "
                           "sampling point, there is no iteration to execute for this idle_tid");
      CodeGen->EmitDebugInfo("cout << \"[\" << start_chunk.first << \">\" << start_chunk.second "
                             "<< \"] No available iteration for \" << idle_tid <<  endl");
      CodeGen->EmitStmt("idle_threads_iterator++");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitLabel("BEGIN_START_POINT_SAMPLE");
#if 0
      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> dist_start_chunk"
                        "(start_chunk.first, start_chunk.second)");
#endif
      string loop_step = Translator->ValueToStringExpr(ParallelLoop->getLoopBound()->StepValue, status);
      CodeGen->EmitCode("if (start_chunk.first == start_chunk.second) {");
      CodeGen->tab_count++;
      if (stoi(loop_step) < 0) {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second");
      } else {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first");
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      if (stoi(loop_step) < 0) {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second + ((rand()%(start_chunk.first-start_chunk.second)+start_chunk.second)-start_chunk.second)/" +
                          loop_step + ")*" + loop_step);
      } else {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first + (((rand()%(start_chunk.second-start_chunk.first)+start_chunk.first)-start_chunk.first)/" +
                          loop_step + ")*" + loop_step);
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      // sample the rest of the loop induction variable
      auto path_riter = pathToParallelTreeRoot.rbegin()+1;
      while (path_riter != pathToParallelTreeRoot.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode "
                             "type");
        string
            induction_expr =
            InductionVarNameTable[LoopInPath->getInductionPhi()],
            loop_init_expr = induction_expr + "_lb",
            loop_final_expr = induction_expr + "_ub", loop_step = "1";

        // need to handle the case where the loop iteration range
        // depends on one of its parent loop
        pair<LoopTNode *, LoopTNode *> IndvarDepResult =
            CheckDependenciesInPath(LoopInPath, pathToParallelTreeRoot);
        string tmp = Translator->ValueToStringExpr(
            LoopInPath->getLoopBound()->InitValue, status);
        if (status == SUCCESS) {
          loop_init_expr = tmp;
        }
        if (IndvarDepResult.first) { // has lower dependency
          if (IndvarDepResult.first == ParallelLoop) {
            ReplaceSubstringWith(loop_init_expr,
                                 InductionVarNameTable[IndvarDepResult.first->getInductionPhi()],
                                 "parallel_loop_start");
          } else {
            ReplaceSubstringWith(loop_init_expr,
                                 InductionVarNameTable[IndvarDepResult.first->getInductionPhi()],
                                 "sample_start_" +
                                 InductionVarNameTable[IndvarDepResult.first->getInductionPhi()]);
          }
        }
        if (LoopInPath->getLoopBound()->FinalValue) {
          tmp = Translator->ValueToStringExpr(
              LoopInPath->getLoopBound()->FinalValue, status);
          if (status == SUCCESS)
            loop_final_expr = tmp;
          if (IndvarDepResult.second) {
            if (IndvarDepResult.second == ParallelLoop) {
              ReplaceSubstringWith(loop_final_expr,
                                   InductionVarNameTable[IndvarDepResult.second->getInductionPhi()],
                                   "parallel_loop_start");
            } else {
              ReplaceSubstringWith(loop_final_expr,
                                   InductionVarNameTable[IndvarDepResult.second->getInductionPhi()],
                                   "sample_start_" +
                                   InductionVarNameTable[IndvarDepResult.second->getInductionPhi()]);
            }
          }
        }
        Translator->ValueToStringExpr(
            LoopInPath->getLoopBound()->StepValue, status);
        if (status == SUCCESS)
          loop_step = Translator->ValueToStringExpr(
              LoopInPath->getLoopBound()->StepValue, status);
        // build the sample bound expression
        // the bound would be
        // [ init_val / step, final_val / step] * step + init_val
        // or
        // [ init_val / step, final_val / step) * step + init_val
        //
        // the position of init_val and final_val depends on the sign
        // of step
        string sample_upper_bound_expr = "";
        sample_upper_bound_expr =
            loop_init_expr + "+" +
            loop_step + "*((" + loop_final_expr + "-" + loop_init_expr + ")"
            + "/" + loop_step;
        switch (LoopInPath->getLoopBound()->Predicate) {
        case llvm::CmpInst::ICMP_UGT:
        case llvm::CmpInst::ICMP_SGT:
        case llvm::CmpInst::ICMP_ULT:
        case llvm::CmpInst::ICMP_SLT:
        case llvm::CmpInst::ICMP_NE:
          sample_upper_bound_expr += "-((" + loop_final_expr + "-" + loop_init_expr + ")%" +
                                     loop_step + "==0)";;
          break;
        default:
          break;
        }
        sample_upper_bound_expr += ")";
        if (path_riter != pathToParallelTreeRoot.rbegin()) {
          CodeGen->EmitCode(
              "if (" + sample_upper_bound_expr + "< 0) { goto BEGIN_START_POINT_SAMPLE; }");
        }
#if 0
        ss << "uniform_int_distribution<mt19937::result_type> dist_start_sample_";
        ss << induction_expr << "(0" << ",";
        ss << sample_upper_bound_expr << ")";
        CodeGen->EmitStmt(ss.str());
#endif
        ss.str(""); // clear the stream to generate other sampling stmts
//                  ss << "int sample_" << induction_expr << " = rand() % ("
//                     << loop_final_expr << "-" << loop_init_expr << ") + "
//                     << loop_step;
#if 0
        ss << "int sample_start_" << induction_expr << " = (dist_start_sample_"
           << induction_expr << "(rng)*(" << loop_step
           << ")+(" << loop_init_expr << "))";
#endif
        ss << "int sample_start_" << induction_expr << " = ("
           << sample_upper_bound_expr << " == 0)?(" << loop_init_expr << "):"
           << "(rand()%(" << sample_upper_bound_expr << "))*(" << loop_step
           << ")+(" << loop_init_expr << ")";
        CodeGen->EmitStmt(ss.str());
        ss.str("");
        path_riter++;
      }
      // we now let the sampler randomly select a RefNode attached
      // inside the parallel loop and assign its reference as the start point
      ss << "vector<string>start_ref_candidates = {";
      set<SPSTNode *> NodesInParallelLoop = G->GetAllNodesInGraph();
      auto it = NodesInParallelLoop.begin();
      while (it != NodesInParallelLoop.end()) {
        if (RefTNode *ref = dynamic_cast<RefTNode *>(*it)) {
          ss << "\"" << SPSNodeNameTable[ref] << "\",";
        }
        it++;
      }
      ss << "}";
      CodeGen->EmitStmt(ss.str());
      ss.str("");
#if 0
      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
                        "dist_start_ref(0, start_ref_candidates.size()-1)");
#endif
      CodeGen->EmitStmt("string start_ref = start_ref_candidates[rand()%start_ref_candidates.size()]");
      CodeGen->EmitDebugInfo("cout << idle_tid << \" starts from reference \" << "
                             "start_ref << endl");
      CodeGen->EmitCode("if (progress[idle_tid]) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("delete progress[idle_tid]");
      CodeGen->EmitStmt("progress[idle_tid] = nullptr");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      ss << "Progress *p = new Progress(start_ref, {";
      path_riter = pathToParallelTreeRoot.rbegin();
      while (path_riter != pathToParallelTreeRoot.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode "
                             "type");
        if (path_riter == pathToParallelTreeRoot.rbegin()) {
          ss << "parallel_loop_start";
        } else {
          ss << "sample_start_"
             << InductionVarNameTable[LoopInPath->getInductionPhi()];
        }
        if (path_riter < pathToParallelTreeRoot.rend()-1)
          ss << ",";
        path_riter++;
      }
      ss << "}, start_chunk)";
      CodeGen->EmitStmt(ss.str());
      ss.str("");
      // it is possible that the TargetRef and the start_ref has different
      // iteration vector size.
      // i.e.
      // for (i = 0; i < N; i++)
      //   A[i]
      //   for (j = 0; j < N; j++)
      //      B[i][j]
      //      C[i][j]
      // if B[i][j] is chosen as the start_ref and A[i] is the TargetRef,
      // we need to randomly choose a j-iteration and push it
      // inside the Progress obj
      it = NodesInParallelLoop.begin();
      while (it != NodesInParallelLoop.end()) {
        if (RefTNode *ref = dynamic_cast<RefTNode *>(*it)) {
          SmallVector<SPSTNode *, 8> dominator;
          G->QueryEngine->GetImmediateCommonLoopDominator(ref, TargetRef, false, dominator);
          AccPath path;
          G->QueryEngine->GetPath(ref, nullptr, path);
          if (path.size() != DistanceToParallel || dominator.front() != path.front()) {
            // DistanceToParall is the size of iteation vector for TargetRef
            // path.size() is the size of iteration vector for the chosen
            // start_ref
            // if DistanceToParallel < path.size(), we need to push
            // if DistanceToParallel > path.size(), we need to pop
            CodeGen->EmitCode("if (start_ref == \""+SPSNodeNameTable[ref]+"\") {");
            CodeGen->tab_count++;
            auto iter = pathToParallelTreeRoot.begin();
            while (*iter != dominator.front() && iter != pathToParallelTreeRoot.end()) {
              CodeGen->EmitStmt("p->iteration.pop_back()");
              iter++;
            }
            auto riter = path.rbegin();
            bool MeetCommonDominator = false;
            while (riter != path.rend()) {
              if (*riter == dominator.front()) {
                MeetCommonDominator = true;
                riter++;
                continue;
              }
              if (*riter != dominator.front() && !MeetCommonDominator) {
                riter++;
                continue;
              } else {
                LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*riter);
                assert(LoopInPath &&
                           "All nodes in the path should be LoopTNode "
                           "type");
                // randomly select a sample from LoopInPath iteration space
                string induction_expr =
                    InductionVarNameTable[LoopInPath->getInductionPhi()],
                    loop_init_expr = induction_expr + "_lb",
                    loop_final_expr = induction_expr + "_ub",
                    loop_step = "1";

                // need to handle the case where the loop iteration range
                // depends on one of its parent loop
                pair<LoopTNode *, LoopTNode *> IndvarDepResult =
                    CheckDependenciesInPath(LoopInPath, pathToParallelTreeRoot);
                string tmp = Translator->ValueToStringExpr(
                    LoopInPath->getLoopBound()->InitValue, status);
                if (status == SUCCESS) {
                  loop_init_expr = tmp;
                }
                if (IndvarDepResult.first) { // has lower dependency
                  if (IndvarDepResult.first == ParallelLoop) {
                    ReplaceSubstringWith(
                        loop_init_expr,
                        InductionVarNameTable[IndvarDepResult.first
                            ->getInductionPhi()],
                        "parallel_loop_start");
                  } else {
                    ReplaceSubstringWith(
                        loop_init_expr,
                        InductionVarNameTable[IndvarDepResult.first
                            ->getInductionPhi()],
                        "sample_start_" +
                        InductionVarNameTable[IndvarDepResult.first
                            ->getInductionPhi()]);
                  }
                }
                if (LoopInPath->getLoopBound()->FinalValue) {
                  tmp = Translator->ValueToStringExpr(
                      LoopInPath->getLoopBound()->FinalValue, status);
                  if (status == SUCCESS)
                    loop_final_expr = tmp;
                  if (IndvarDepResult.second) {
                    if (IndvarDepResult.second == ParallelLoop) {
                      ReplaceSubstringWith(
                          loop_final_expr,
                          InductionVarNameTable[IndvarDepResult.second
                              ->getInductionPhi()],
                          "parallel_loop_start");
                    } else {
                      ReplaceSubstringWith(
                          loop_final_expr,
                          InductionVarNameTable[IndvarDepResult.second
                              ->getInductionPhi()],
                          "sample_start_" +
                          InductionVarNameTable[IndvarDepResult.second
                              ->getInductionPhi()]);
                    }
                  }
                }
                Translator->ValueToStringExpr(
                    LoopInPath->getLoopBound()->StepValue, status);
                if (status == SUCCESS)
                  loop_step = Translator->ValueToStringExpr(
                      LoopInPath->getLoopBound()->StepValue, status);
                // build the sample bound expression
                // the bound would be
                // [ init_val / step, final_val / step] * step + init_val
                // or
                // [ init_val / step, final_val / step) * step + init_val
                //
                // the position of init_val and final_val depends on the sign
                // of step
                string sample_upper_bound_expr = "";
                sample_upper_bound_expr =
                    loop_init_expr + "+" + loop_step + "*((" + loop_final_expr +
                    "-" + loop_init_expr + ")" + "/" + loop_step;
                switch (LoopInPath->getLoopBound()->Predicate) {
                case llvm::CmpInst::ICMP_UGT:
                case llvm::CmpInst::ICMP_SGT:
                case llvm::CmpInst::ICMP_ULT:
                case llvm::CmpInst::ICMP_SLT:
                case llvm::CmpInst::ICMP_NE:
                  sample_upper_bound_expr += "-((" + loop_final_expr + "-" +
                                             loop_init_expr + ")%" + loop_step +
                                             "==0)";
                  ;
                  break;
                default:
                  break;
                }
                sample_upper_bound_expr += ")";
                if (path_riter != pathToParallelTreeRoot.rbegin()) {
                  CodeGen->EmitCode("if (" + sample_upper_bound_expr +
                                    "< 0) { goto BEGIN_START_POINT_SAMPLE; }");
                }
#if 0
                ss << "uniform_int_distribution<mt19937::result_type> dist_start_sample_";
                ss << induction_expr << "(0"
                   << ",";
                ss << sample_upper_bound_expr << ")";
                CodeGen->EmitStmt(ss.str());
#endif
                ss.str(""); // clear the stream to generate other sampling stmts
                //                  ss << "int sample_" << induction_expr << " = rand() % ("
                //                     << loop_final_expr << "-" << loop_init_expr << ") + "
                //                     << loop_step;
                ss << "int sample_start_" << induction_expr << " = ("
                   << sample_upper_bound_expr << " == 0)?(" << loop_init_expr << "):"
                   << "(rand()%(" << sample_upper_bound_expr << "))*(" << loop_step
                   << ")+(" << loop_init_expr << ")";
                CodeGen->EmitStmt(ss.str());
                ss.str("");
                CodeGen->EmitStmt("p->iteration.push_back(sample_start_" +
                                  induction_expr + ")");
              }
              riter++;
            }
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
          }
        }
        it++;
      }
#if 0
      CodeGen->EmitStmt("per_thread_start_point[idle_tid] = p");
#endif
      CodeGen->EmitStmt("progress[idle_tid] = p");
      CodeGen->EmitDebugInfo("cout << \"[\" << idle_tid << \"] starts from \" << p->toString() << endl");
      CodeGen->EmitStmt("idle_threads_iterator = idle_threads.erase(idle_threads_iterator)");
      CodeGen->EmitDebugInfo("cout << \"Move \" << idle_tid << \" to worker_threads\" << endl");
      CodeGen->EmitStmt("worker_threads.push_back(idle_tid)");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of dispatcher.hasNextStaticChunk(idle_tid) check");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("Chunk start_chunk = dispatcher.getStaticStartChunk("
                        "start_iteration.ivs["+to_string(DistanceToTop)+"], idle_tid)");
      CodeGen->EmitCode("if (progress[idle_tid]) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("delete progress[idle_tid]");
      CodeGen->EmitStmt("progress[idle_tid] = nullptr");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      ss << "Progress *p = new Progress(\"" << SPSNodeNameTable[TargetRef] << "\",{";
      path_riter = pathToParallelTreeRoot.rbegin();
      int i = 0;
      while (path_riter != pathToParallelTreeRoot.rend()) {
        ss << "start_iteration.ivs[" << DistanceToTop+i << "]";
        if (path_riter != pathToParallelTreeRoot.rend()-1)
          ss << ",";
        path_riter++;
        i++;
      }
      ss << "},start_chunk)";
      CodeGen->EmitStmt(ss.str());
      ss.str("");
#if 0
      CodeGen->EmitStmt("per_thread_start_point[idle_tid] = p");
#endif
      CodeGen->EmitStmt("progress[idle_tid] = p");
      CodeGen->EmitDebugInfo("cout << \"[\" << idle_tid << \"] starts from \" << progress[idle_tid]->toString() << endl");
      CodeGen->EmitStmt("idle_threads_iterator = idle_threads.erase(idle_threads_iterator)");
      CodeGen->EmitDebugInfo("cout << \"Move \" << idle_tid << \" to worker_threads\" << endl");
      CodeGen->EmitStmt("worker_threads.push_back(idle_tid)");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitStmt("idle_threads_iterator++");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of tid_to_run traversal");
#if 0
      CodeGen->EmitCode("for (tid_to_run = 0; tid_to_run < THREAD_NUM; tid_to_run++) {");
      for (tid_to_run = 0; tid_to_run < THREAD_NUM; tid_to_run++) {
        if (tid_to_run !=
            dispatcher.getStaticTid(start_iteration.ivs[DistanceToTop])) {
          int start_cid = dispatcher.getStaticChunkID(start_iteration.ivs[DistanceToTop]);
          Chunk start_chunk = dispatcher.getStaticChunkWithCid(start_cid, tid_to_run);
          uniform_int_distribution<mt19937::result_type> dist_chunk(start_chunk.first, start_chunk.second);
          int parallel_loop_start = dist_chunk(rng);
          // sample the rest of the loop induction variable

          Progress p("ref randomly chosen by the compiler", {parallel_loop_start, ,});
        } else {
          Progress p('target ref name', start_iteration.ivs);
        }
      }
#endif
    } else if (schedule == DYNAMIC) {
      string loop_step = Translator->ValueToStringExpr(ParallelLoop->getLoopBound()->StepValue, status);
      // the following code will be generated by the SamplerCodeGen
      CodeGen->EmitStmt("int next_chunk_ptr = 0, prev_chunk_ptr = 0");
      CodeGen->EmitStmt("vector<Chunk> next_chunks, prev_chunks");
      CodeGen->EmitStmt("Chunk start_chunk = dispatcher.getStartChunk(start_iteration.ivs["
                        +to_string(DistanceToTop)+"])");
      CodeGen->EmitComment("we get THREAD_NUM-1 next chunks and "
                           "THREAD_NUM-1 previous chunks");
      CodeGen->EmitStmt("dispatcher.getPrevKChunksFrom(THREAD_NUM-1, "
                        "start_chunk, prev_chunks)");
      CodeGen->EmitStmt("dispatcher.getNextKChunksFrom(THREAD_NUM-1, "
                        "start_chunk, next_chunks)");
      CodeGen->EmitStmt("auto iter = idle_threads.begin()");
      CodeGen->EmitStmt("int first_idle_tid = *iter");
      CodeGen->EmitCode("while (iter != idle_threads.end()) {");
      CodeGen->tab_count++;
      // in the dynamic scheduling, we assign the start_iteration.ivs as the
      // start point of the first tid in the random_shuffle of idle_threads
      // for the rest, we examine is previous chunk of its next chunk with 50%
      // probability
      CodeGen->EmitCode("if (*iter == first_idle_tid) {");
      CodeGen->tab_count++;
      CodeGen->EmitCode("if (progress[first_idle_tid]) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("progress[first_idle_tid]->ref = \"" +  SPSNodeNameTable[TargetRef] + "\"");
      auto path_riter = pathToParallelTreeRoot.rbegin();
      int i = 0;
      ss << "{";
      while (path_riter != pathToParallelTreeRoot.rend()) {
        ss << "start_iteration.ivs[" << DistanceToTop+i << "]";
        if (path_riter < pathToParallelTreeRoot.rend()-1)
          ss << ",";
        path_riter++;
        i++;
      }
      ss << "}";
      CodeGen->EmitStmt("progress[first_idle_tid]->iteration = " + ss.str());
      CodeGen->EmitStmt("progress[first_idle_tid]->chunk = start_chunk");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      ss << ",start_chunk)";
      CodeGen->EmitStmt("Progress *p = new Progress(\""+SPSNodeNameTable[TargetRef]+"\","+ss.str());
      ss.str("");
#if 0
      CodeGen->EmitStmt("per_thread_start_point[*iter] = p");
#endif
      CodeGen->EmitStmt("progress[first_idle_tid] = p");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitDebugInfo("cout << \"[\" << first_idle_tid << \"] starts from \" << progress[first_idle_tid]->toString() << endl");
      CodeGen->EmitStmt("iter = idle_threads.erase(iter)");
      CodeGen->EmitDebugInfo("cout << \"Move \" << first_idle_tid << \" to worker_threads\" << endl");
      CodeGen->EmitStmt("worker_threads.push_back(first_idle_tid)");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      CodeGen->EmitComment("We first init the start_chunk with a invalid value, "
                           "since it is possible that no start_chunk found");
      CodeGen->EmitStmt("start_chunk = make_pair(0, -1)");
      CodeGen->EmitStmt("int parallel_loop_start = -1");
#if 0
      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
                        "dist_direction(1, 100)");
      CodeGen->EmitStmt("int direction = dist_direction(rng)");
#endif
      CodeGen->EmitStmt("int direction = rand()%100");
      CodeGen->EmitCode("if (direction < 50) {");
      CodeGen->tab_count++;
      CodeGen->EmitCode("if (next_chunk_ptr < next_chunks.size() || prev_chunks.empty()) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("start_chunk = next_chunks[next_chunk_ptr]");
      CodeGen->EmitStmt("next_chunk_ptr++");
      // here we need to update the bound of the next available chunk to be
      // assigned, since its current "next chunk" has already been chosen as the
      // start point of another thread
      CodeGen->EmitStmt("dispatcher.moveToNextChunk()");
//      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
//                        "dist_chunk(start_chunk.first, start_chunk.second)");
//      if (stoi(loop_step) < 0) {
//        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second + ((dist_chunk(rng)-start_chunk.second)/" +
//                          loop_step + ")*" + loop_step);
//      } else {
//        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first + ((dist_chunk(rng)-start_chunk.first)/" +
//                          loop_step + ")*" + loop_step);
//      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      CodeGen->EmitComment("no next_chunk available, no start point of this thread");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      CodeGen->EmitCode("if (prev_chunk_ptr < prev_chunks.size() || next_chunks.empty()) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("start_chunk = prev_chunks[prev_chunk_ptr]");
      CodeGen->EmitStmt("prev_chunk_ptr++");
//      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
//                        "dist_chunk(start_chunk.first, start_chunk.second)");
//      if (stoi(loop_step) < 0) {
//        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second + ((dist_chunk(rng)-start_chunk.second)/" +
//                          loop_step + ")*" + loop_step);
//      } else {
//        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first + ((dist_chunk(rng)-start_chunk.first)/" +
//                          loop_step + ")*" + loop_step);
//      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      CodeGen->EmitComment("no prev_chunk available, no start point of this thread");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      CodeGen->EmitCode("if (start_chunk.first <= start_chunk.second) {");
      CodeGen->tab_count++;
      CodeGen->EmitLabel("BEGIN_START_POINT_SAMPLE");
#if 0
      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
                        "dist_start_chunk(start_chunk.first, start_chunk.second)");
#endif
      CodeGen->EmitCode("if (start_chunk.first == start_chunk.second) {");
      CodeGen->tab_count++;
      if (stoi(loop_step) < 0) {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second");
      } else {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first");
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("} else {");
      CodeGen->tab_count++;
      if (stoi(loop_step) < 0) {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.second + ((rand()%(start_chunk.first-start_chunk.second)+start_chunk.second)-start_chunk.second)/" +
                          loop_step + ")*" + loop_step);
      } else {
        CodeGen->EmitStmt("parallel_loop_start = start_chunk.first + (((rand()%(start_chunk.second-start_chunk.first)+start_chunk.first)-start_chunk.first)/" +
                          loop_step + ")*" + loop_step);
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      if (dyn_cast<ConstantInt>(ParallelLoop->getLoopBound()->StepValue)->isNegative()) {
        CodeGen->EmitCode("if (parallel_loop_start > 0 && parallel_loop_start >= start_chunk.first) {");
      } else {
        CodeGen->EmitCode("if (parallel_loop_start > 0 && parallel_loop_start <= start_chunk.second) {");
      }
      CodeGen->tab_count++;
      // sample the rest of the loop induction variable
      path_riter = pathToParallelTreeRoot.rbegin()+1;
      while (path_riter != pathToParallelTreeRoot.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode "
                             "type");
        TranslateStatus status = SUCCESS;
        string
            induction_expr =
            InductionVarNameTable[LoopInPath->getInductionPhi()],
            loop_init_expr = induction_expr + "_lb",
            loop_final_expr = induction_expr + "_ub", loop_step = "1";

        // need to handle the case where the loop iteration range
        // depends on one of its parent loop
        pair<LoopTNode *, LoopTNode *> IndvarDepResult =
            CheckDependenciesInPath(LoopInPath, pathToParallelTreeRoot);
        string tmp = Translator->ValueToStringExpr(
            LoopInPath->getLoopBound()->InitValue, status);
        if (status == SUCCESS) {
          loop_init_expr = Translator->ValueToStringExpr(
              LoopInPath->getLoopBound()->InitValue, status);
        }
        if (IndvarDepResult.first) { // has lower dependency
          if (IndvarDepResult.first == ParallelLoop) {
            ReplaceSubstringWith(loop_init_expr,
                                 InductionVarNameTable[IndvarDepResult.first->getInductionPhi()],
                                 "parallel_loop_start");
          } else {
            ReplaceSubstringWith(loop_init_expr,
                                 InductionVarNameTable[IndvarDepResult.first->getInductionPhi()],
                                 "sample_start_" +
                                 InductionVarNameTable[IndvarDepResult.first->getInductionPhi()]);
          }
        }
        if (LoopInPath->getLoopBound()->FinalValue) {
          string tmp = Translator->ValueToStringExpr(
              LoopInPath->getLoopBound()->FinalValue, status);
          if (status == SUCCESS)
            loop_final_expr = tmp;
          if (IndvarDepResult.second) {
            if (IndvarDepResult.second == ParallelLoop) {
              ReplaceSubstringWith(loop_final_expr,
                                   InductionVarNameTable[IndvarDepResult.second->getInductionPhi()],
                                   "parallel_loop_start");
            } else {
              ReplaceSubstringWith(loop_final_expr,
                                   InductionVarNameTable[IndvarDepResult.second->getInductionPhi()],
                                   "sample_start_" +
                                   InductionVarNameTable[IndvarDepResult.second->getInductionPhi()]);
            }
          }
        }
        Translator->ValueToStringExpr(
            LoopInPath->getLoopBound()->StepValue, status);
        if (status == SUCCESS)
          loop_step = Translator->ValueToStringExpr(
              LoopInPath->getLoopBound()->StepValue, status);
        // build the sample bound expression
        // the bound would be
        // [ init_val / step, final_val / step] * step + init_val
        // or
        // [ init_val / step, final_val / step) * step + init_val
        //
        // the position of init_val and final_val depends on the sign
        // of step
        string sample_upper_bound_expr = "";
        sample_upper_bound_expr = "((" + loop_final_expr + "-" + loop_init_expr + ")"
                                  + "/" + loop_step;
        switch (LoopInPath->getLoopBound()->Predicate) {
        case llvm::CmpInst::ICMP_UGT:
        case llvm::CmpInst::ICMP_SGT:
        case llvm::CmpInst::ICMP_ULT:
        case llvm::CmpInst::ICMP_SLT:
        case llvm::CmpInst::ICMP_NE:
          sample_upper_bound_expr += "-((" + loop_final_expr + "-" + loop_init_expr + ")%" +
                                     loop_step + "==0)";
          break;
        default:
          break;
        }
        sample_upper_bound_expr += ")";
        if (path_riter != pathToParallelTreeRoot.rbegin()) {
          CodeGen->EmitCode(
              "if ("+sample_upper_bound_expr + "< 0) { goto BEGIN_START_POINT_SAMPLE; }");
        }
#if 0
        ss << "uniform_int_distribution<mt19937::result_type> dist_start_sample_";
        ss << induction_expr << "(0," << sample_upper_bound_expr << ")";
        CodeGen->EmitStmt(ss.str());
        ss.str(""); // clear the stream to generate other sampling stmts
//                  ss << "int sample_" << induction_expr << " = rand() % ("
//                     << loop_final_expr << "-" << loop_init_expr << ") + "
//                     << loop_step;
        ss << "int sample_start_" << induction_expr << " = (dist_start_sample_"
           << induction_expr << "(rng)*(" << loop_step
           << ")+(" << loop_init_expr << "))";
#endif
        ss << "int sample_start_" << induction_expr << " = ("
           << sample_upper_bound_expr << " == 0)?(" << loop_init_expr << "):"
           << "(rand()%(" << sample_upper_bound_expr << "))*(" << loop_step
           << ")+(" << loop_init_expr << ")";
        CodeGen->EmitStmt(ss.str());
        ss.str("");
        path_riter++;
      }
      // we now let the sampler randomly select a RefNode attached
      // inside the parallel loop and assign its reference as the start point
      ss << "vector<string>start_ref_candidates = {";
      set<SPSTNode *> NodesInParallelLoop = G->GetAllNodesInGraph();
      auto it = NodesInParallelLoop.begin();
      while (it != NodesInParallelLoop.end()) {
        if (RefTNode *ref = dynamic_cast<RefTNode *>(*it)) {
          ss << "\"" << SPSNodeNameTable[ref] << "\",";
        }
        it++;
      }
      ss << "}";
      CodeGen->EmitStmt(ss.str());
      ss.str("");
#if 0
      CodeGen->EmitStmt("uniform_int_distribution<mt19937::result_type> "
                        "dist_start_ref(0, start_ref_candidates.size()-1)");
#endif
      CodeGen->EmitStmt("string start_ref = start_ref_candidates[rand()%start_ref_candidates.size()]");
      CodeGen->EmitDebugInfo("cout << *iter << \" starts from reference \" << "
                             "start_ref << endl");
      // Generate code that sample the rest of the iteration
      CodeGen->EmitCode("if (progress[*iter]) {");
      CodeGen->tab_count++;
      CodeGen->EmitStmt("delete progress[*iter]");
      CodeGen->EmitStmt("progress[*iter] = nullptr");
      CodeGen->tab_count--;
      CodeGen->EmitCode("}");
      ss << "Progress *p = new Progress(start_ref, {";
      path_riter = pathToParallelTreeRoot.rbegin();
      while (path_riter != pathToParallelTreeRoot.rend()) {
        LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
        assert(LoopInPath && "All nodes in the path should be LoopTNode type");
        if (LoopInPath == ParallelLoop) { // this is the parallel loop
          ss << "parallel_loop_start";
        } else if (LoopInPath->getLoopLevel() <
                   ParallelLoop->getLoopLevel()) { // this is the parent of the parallel loop
          ss << Translator->ValueToStringExpr(LoopInPath->getInductionPhi(),
                                              status);
        } else { // this is the loop inside the parallel loop
          ss << "sample_start_"
             << InductionVarNameTable[LoopInPath->getInductionPhi()];
        }
        if (path_riter != pathToTreeRoot.rend() - 1)
          ss << ", ";
        path_riter++;
      }
      ss << "}, start_chunk)";
      CodeGen->EmitStmt(ss.str());
      ss.str("");
      // it is possible that the TargetRef and the start_ref has different
      // iteration vector size.
      // i.e.
      // for (i = 0; i < N; i++)
      //   A[i]
      //   for (j = 0; j < N; j++)
      //      B[i][j]
      //      C[i][j]
      // if B[i][j] is chosen as the start_ref and A[i] is the TargetRef,
      // we need to randomly choose a j-iteration and push it
      // inside the Progress obj
      it = NodesInParallelLoop.begin();
      while (it != NodesInParallelLoop.end()) {
        if (RefTNode *ref = dynamic_cast<RefTNode *>(*it)) {
          SmallVector<SPSTNode *, 8> dominator;
          G->QueryEngine->GetImmediateCommonLoopDominator(ref, TargetRef, false, dominator);
          AccPath path;
          G->QueryEngine->GetPath(ref, nullptr, path);
          if (path.size() != DistanceToParallel || dominator.front() != path.front()) {
            // DistanceToParall is the size of iteation vector for TargetRef
            // path.size() is the size of iteration vector for the chosen
            // start_ref
            // if DistanceToParallel < path.size(), we need to push
            // if DistanceToParallel > path.size(), we need to pop
            CodeGen->EmitCode("if (start_ref == \""+SPSNodeNameTable[ref]+"\") {");
            CodeGen->tab_count++;
            auto iter = pathToParallelTreeRoot.begin();
            while (*iter != dominator.front() && iter != pathToParallelTreeRoot.end()) {
              CodeGen->EmitStmt("p->iteration.pop_back()");
              iter++;
            }
            auto riter = path.rbegin();
            bool MeetCommonDominator = false;
            while (riter != path.rend()) {
              if (*riter == dominator.front()) {
                MeetCommonDominator = true;
                riter++;
                continue;
              }
              if (*riter != dominator.front() && !MeetCommonDominator) {
                riter++;
                continue;
              } else {
                LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*riter);
                assert(LoopInPath &&
                           "All nodes in the path should be LoopTNode "
                           "type");
                // randomly select a sample from LoopInPath iteration space
                string induction_expr =
                    InductionVarNameTable[LoopInPath->getInductionPhi()],
                    loop_init_expr = induction_expr + "_lb",
                    loop_final_expr = induction_expr + "_ub",
                    loop_step = "1";

                // need to handle the case where the loop iteration range
                // depends on one of its parent loop
                pair<LoopTNode *, LoopTNode *> IndvarDepResult =
                    CheckDependenciesInPath(LoopInPath, pathToParallelTreeRoot);
                string tmp = Translator->ValueToStringExpr(
                    LoopInPath->getLoopBound()->InitValue, status);
                if (status == SUCCESS) {
                  loop_init_expr = tmp;
                }
                if (IndvarDepResult.first) { // has lower dependency
                  if (IndvarDepResult.first == ParallelLoop) {
                    ReplaceSubstringWith(
                        loop_init_expr,
                        InductionVarNameTable[IndvarDepResult.first
                            ->getInductionPhi()],
                        "parallel_loop_start");
                  } else {
                    ReplaceSubstringWith(
                        loop_init_expr,
                        InductionVarNameTable[IndvarDepResult.first
                            ->getInductionPhi()],
                        "sample_start_" +
                        InductionVarNameTable[IndvarDepResult.first
                            ->getInductionPhi()]);
                  }
                }
                if (LoopInPath->getLoopBound()->FinalValue) {
                  tmp = Translator->ValueToStringExpr(
                      LoopInPath->getLoopBound()->FinalValue, status);
                  if (status == SUCCESS)
                    loop_final_expr = tmp;
                  if (IndvarDepResult.second) {
                    if (IndvarDepResult.second == ParallelLoop) {
                      ReplaceSubstringWith(
                          loop_final_expr,
                          InductionVarNameTable[IndvarDepResult.second
                              ->getInductionPhi()],
                          "parallel_loop_start");
                    } else {
                      ReplaceSubstringWith(
                          loop_final_expr,
                          InductionVarNameTable[IndvarDepResult.second
                              ->getInductionPhi()],
                          "sample_start_" +
                          InductionVarNameTable[IndvarDepResult.second
                              ->getInductionPhi()]);
                    }
                  }
                }
                Translator->ValueToStringExpr(
                    LoopInPath->getLoopBound()->StepValue, status);
                if (status == SUCCESS)
                  loop_step = Translator->ValueToStringExpr(
                      LoopInPath->getLoopBound()->StepValue, status);
                // build the sample bound expression
                // the bound would be
                // [ init_val / step, final_val / step] * step + init_val
                // or
                // [ init_val / step, final_val / step) * step + init_val
                //
                // the position of init_val and final_val depends on the sign
                // of step
                string sample_upper_bound_expr = "";
                sample_upper_bound_expr =
                    loop_init_expr + "+" + loop_step + "*((" + loop_final_expr +
                    "-" + loop_init_expr + ")" + "/" + loop_step;
                switch (LoopInPath->getLoopBound()->Predicate) {
                case llvm::CmpInst::ICMP_UGT:
                case llvm::CmpInst::ICMP_SGT:
                case llvm::CmpInst::ICMP_ULT:
                case llvm::CmpInst::ICMP_SLT:
                case llvm::CmpInst::ICMP_NE:
                  sample_upper_bound_expr += "-((" + loop_final_expr + "-" +
                                             loop_init_expr + ")%" + loop_step +
                                             "==0)";
                  ;
                  break;
                default:
                  break;
                }
                sample_upper_bound_expr += ")";
                if (path_riter != pathToParallelTreeRoot.rbegin()) {
                  CodeGen->EmitCode("if (" + sample_upper_bound_expr +
                                    "< 0) { goto BEGIN_START_POINT_SAMPLE; }");
                }
                ss << "uniform_int_distribution<mt19937::result_type> dist_start_sample_";
                ss << induction_expr << "(0"
                   << ",";
                ss << sample_upper_bound_expr << ")";
                CodeGen->EmitStmt(ss.str());
                ss.str(""); // clear the stream to generate other sampling stmts
                //                  ss << "int sample_" << induction_expr << " = rand() % ("
                //                     << loop_final_expr << "-" << loop_init_expr << ") + "
                //                     << loop_step;
#if 0
                ss << "int sample_start_" << induction_expr
                   << " = (dist_start_sample_" << induction_expr << "(rng)*("
                   << loop_step << ")+(" << loop_init_expr << "))";
#endif
                ss << "int sample_start_" << induction_expr << " = ("
                   << sample_upper_bound_expr << " == 0)?(" << loop_init_expr << "):"
                   << "(rand()%(" << sample_upper_bound_expr << "))*(" << loop_step
                   << ")+(" << loop_init_expr << ")";
                CodeGen->EmitStmt(ss.str());
                ss.str("");
                CodeGen->EmitStmt("p->iteration.push_back(sample_start_" +
                                  induction_expr + ")");
              }
              riter++;
            }
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
          }
        }
        it++;
      }
#if 0
      CodeGen->EmitStmt("per_thread_start_point[*iter] = p");
#endif
      CodeGen->EmitStmt("progress[*iter] = p");
      CodeGen->EmitDebugInfo("cout << \"[\" << *iter << \"] starts from \" << p->toString() << endl");
      CodeGen->EmitDebugInfo("cout << \"Move \" << *iter << \" to worker_threads\" << endl");
      CodeGen->EmitStmt("worker_threads.push_back(*iter)");
      CodeGen->EmitStmt("iter = idle_threads.erase(iter)");
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of if (parallel_loop_start > 0)");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of start_chunk validation");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // if (iter != first_idle_tid)");
      CodeGen->EmitStmt("iter++");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of idle_threads traversal");
#if 0
      CodeGen->EmitStmt("random_shuffle(idle_threads.begin(), "
                        "idle_threads.end(), [](int n) { return rand() % n; });");
      // in the dynamic scheduling, we assign the start_iteration.ivs as the
      // start point of the first tid in the random_shuffle of idle_threads
      // for the rest, we examine is previous chunk of its next chunk with 50%
      // probability
      auto iter = idle_threads.begin();
      int next_chunk_ptr = 0, prev_chunk_ptr;
      while (iter != idle_threads.end()) {
        if (iter == idle_threads.begin()) {
          Progress p("target ref id", start_iteration);
        } else {
          vector<Chunk> next_chunks, prev_chunks;
          Chunk start_chunk = dispatcher.getStartChunk(start_iteration.ivs[DistanceToTop]);
          // we get THREAD_NUM-1 next chunks and THREAD_NUM-1 previous chunks
          dispatcher.getKPrevChunksFrom(THREAD_NUM-1, start_chunk, prev_chunks);
          dispatcher.getKNextChunksFrom(THREAD_NUM-1, start_chunk, next_chunks);
          int parallel_loop_start = 0;
          if (/* next chunk */) {
            start_chunk = next_chunks[next_chunk_ptr];
            next_chunk_ptr++;
            uniform_int_distribution<mt19937::result_type> dist_chunk(
                start_chunk.first, start_chunk.second);
            parallel_loop_start = dist_chunk(rng);
          } else if (/* prev chunk */) {
            start_chunk = prev_chunks[prev_chunk_ptr];
            prev_chunk_ptr++;
            uniform_int_distribution<mt19937::result_type> dist_chunk(
                start_chunk.first, start_chunk.second);
            parallel_loop_start = dist_chunk(rng);
          }
          // sample the rest of the loop induction variable
          Progress p("ref randomly chosen by the compiler", {parallel_loop_start, , });
        }
        iter++;
      }
#endif
    }
  }
}

void ModelCodeGenProWrapperPass::PerRefParallelSamplerBranchBodyGenImpl(DummyTNode *Branch,
                                                                        Interleaving type,
                                                                        SPSTNode *MergePoint,
                                                                        RefTNode *TargetRef)
{
// TODO: we assume now only branch and ref can inside a if-condition
  //  so all node inside a branch has a name in SPSNodeNameTable
  stringstream codess;
  auto branchIter = Branch->neighbors.begin();
  while (branchIter != Branch->neighbors.end()) {
    if (RefTNode *RefNode = dynamic_cast<RefTNode *>(*branchIter)) {
      codess << "if (progress[tid_to_run]->ref == \"" << SPSNodeNameTable[RefNode] << "\") {";
      CodeGen->EmitCode(codess.str());
      codess.str("");
      CodeGen->tab_count++;
      CodeGen->EmitCode("if (start_reuse_search) {");
      CodeGen->tab_count++;
      if (QueryEngine->areAccessToSameArray(RefNode, TargetRef)) {
        // visit this access here if the RefNode access the
        // same array as the TargetRef
        vector<string> params = EmitRefNodeAccessExpr(RefNode, true);
        if (!params.empty()) {
          AccPath path;
          QueryEngine->GetPath(RefNode, nullptr, path);
          if (path.empty() || SPSNodeNameTable[RefNode] != SPSNodeNameTable[TargetRef] ) {
//            codess << "Iteration *access = nullptr";
          } else {
            codess << "Iteration *access = new Iteration(\""
                   << SPSNodeNameTable[RefNode] << "\", {";
            auto pathIter = path.rbegin();
            TranslateStatus status = SUCCESS;
            while (pathIter != path.rend()) {
              LoopTNode *NestLoop = dynamic_cast<LoopTNode *>(*pathIter);
              assert(NestLoop &&
                         "All node in a path should be an instance of LoopTNode");
              string nest_loop_induction = ReplaceInductionVarExprByType(
                  NestLoop->getInductionPhi(), OMP_PARALLEL, status);
              codess << nest_loop_induction;
              if (pathIter + 1 != path.rend())
                codess << ",";
              pathIter++;
            }
            codess << "})";
            CodeGen->EmitStmt(codess.str());
            codess.str("");
          }
          CodeGen->EmitStmt("string array = " + params[0]);
          CodeGen->EmitStmt("string refname = \"" + SPSNodeNameTable[RefNode] + "\"");
//          CodeGen->EmitStmt("subscripts = " + params[1]);
          codess << "addr = GetAddress_" << SPSNodeNameTable[RefNode] << "(";
          for (unsigned i = 0; i < RefNode->getSubscripts().size(); i++) {
//            codess << "subscripts[" << i << "]";
            codess << EmitRefNodeAccessExprAtIdx(RefNode->getSubscripts()[i], true);
            if (i < RefNode->getSubscripts().size()-1)
              codess << ",";
          }
          codess << ")";
          CodeGen->EmitStmt(codess.str());
          codess.str("");
          if (EnableSampling && SamplingMethod == RANDOM_START) {
            CodeGen->EmitStmt("bool isSample = false");
            if (RefNode == TargetRef) {
              CodeGen->EmitDebugInfo("cout << access->toString() << \" @ \" << addr << endl");
              CodeGen->EmitCode(
                  "if (access->compare(start_iteration) == 0) {");
              CodeGen->tab_count++;
              CodeGen->EmitDebugInfo("cout << \"Meet the start sample \" << access->toString() << endl");
              CodeGen->EmitStmt("isSample = true");
              CodeGen->tab_count--;
              CodeGen->EmitCode("} else if ((!samples.empty() && access->compare(samples.top()) == 0) "
                                "|| (sample_names.find(access->toAddrString()) != sample_names.end())) {");
              CodeGen->tab_count++;
              CodeGen->EmitDebugInfo(
                  "cout << \"Meet a new sample \" << access->toString() << \" while searching reuses\" << endl");
              CodeGen->EmitStmt("traversing_samples++");
              CodeGen->EmitCode("if (!samples.empty() && access->compare(samples.top()) == 0) {");
              CodeGen->tab_count++;
              CodeGen->EmitStmt("samples.pop()");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
              CodeGen->EmitStmt("isSample = true");
              CodeGen->tab_count--;
              CodeGen->EmitCode("}");
            }
          }
          CodeGen->EmitCode("if (LAT.find(addr) != LAT.end()) {");
          CodeGen->tab_count++;
          CodeGen->EmitStmt("long reuse = count - LAT[addr]");
          if (EnableParallelOpt) {
            CodeGen->EmitFunctionCall("pluss_parallel_histogram_update", {"histogram", "reuse", "1"});
          } else {
            CodeGen->EmitFunctionCall("pluss_histogram_update", {"reuse", "1."});
          }
          CodeGen->EmitStmt("Iteration *src = LATSampleIterMap[LAT[addr]]");
          if (EnableSampling && SamplingMethod == RANDOM_START) {
            if (TargetRef == RefNode) {
              CodeGen->EmitDebugInfo(
                  "cout << \"[\" << reuse << \"] \" << src->toString() << \" -> \" << access->toString() << endl");
            }
#if 0
            CodeGen->EmitStmt("samples.erase(*src)");
#endif
            CodeGen->EmitStmt("traversing_samples--");
            CodeGen->EmitComment("stop traversing if reuse of all samples"
                                 "are found");
            CodeGen->EmitDebugInfo("if (samples.empty() && traversing_samples == 0) { cout << \"[\" << reuse << \"] for last sample \" << src->toString() << endl; }");
            CodeGen->EmitCode("if (samples.empty() && traversing_samples == 0) { goto END_SAMPLE; }");
            CodeGen->EmitCode("if (traversing_samples == 0) { ");
            CodeGen->tab_count++;
            CodeGen->EmitDebugInfo("cout << \"delete sample \" << src->toString() << \", active:\" << traversing_samples << \", remain:\" << samples.size() << endl");
            CodeGen->EmitStmt("delete src");
            CodeGen->EmitStmt("LATSampleIterMap.erase(LAT[addr])");
            CodeGen->EmitStmt("LAT.erase(addr)");
            CodeGen->EmitComment("Here we examine if there is an out-of-order effect.");
            CodeGen->EmitComment("if the next sample we should jump has been traversed before, we will pop this sample directly.");
            CodeGen->EmitComment("It is safe to call samples.top() once, since when entering here, 'samples' queue is not empty()");
            CodeGen->EmitCode("if (samples_meet.size() >= samples.size()) { goto END_SAMPLE; }");
            CodeGen->EmitStmt("Iteration next = samples.top()");
            CodeGen->EmitCode("while(samples_meet.find(next.toAddrString()) != samples_meet.end()) {");
            CodeGen->tab_count++;
            CodeGen->EmitDebugInfo("cout << \"Skip \" << next.toString() << \" because we met this sample already \" << endl");
            CodeGen->EmitStmt("samples.pop()");
            CodeGen->EmitComment("All samples has been traversed, no need to jump");
            CodeGen->EmitCode("if (samples.empty()) { break; }");
            CodeGen->EmitStmt("next = samples.top()");
            CodeGen->tab_count--;
            CodeGen->EmitCode("} // end of out-of-order check");
            CodeGen->EmitCode("if (!samples.empty()) {");
            CodeGen->tab_count++;
            CodeGen->EmitHashMacro("#if defined(DEBUG)");
            CodeGen->EmitStmt("next = samples.top()");
            CodeGen->EmitStmt("cout << \"Jump to next sample \" << next.toString() << endl");
            CodeGen->EmitHashMacro("#endif");
            CodeGen->EmitStmt("goto START_SAMPLE_"+SPSNodeNameTable[TargetRef]);
            CodeGen->tab_count--;
            CodeGen->EmitCode("} else { goto END_SAMPLE; }");
            CodeGen->tab_count--;
            CodeGen->EmitCode("} // end of if (traversing_samples == 0)");
            CodeGen->tab_count--;
            CodeGen->EmitCode("}");
          }
          CodeGen->EmitDebugInfo("cout << \"delete sample \" << src->toString() << \", active:\" << traversing_samples << \", remain:\" << samples.size() << endl");
          CodeGen->EmitStmt("delete src");
          CodeGen->EmitStmt("LATSampleIterMap.erase(LAT[addr])");
          CodeGen->EmitStmt("LAT.erase(addr)");
          CodeGen->tab_count--;
          CodeGen->EmitCode("} end of if (LAT.find(addr) != LAT.end())");
          if (RefNode == TargetRef) {
            if (EnableSampling && SamplingMethod == RANDOM_START) {
              CodeGen->EmitCode(
                  "if (isSample) {");
              CodeGen->tab_count++;
              CodeGen->EmitStmt("LAT[addr] = count");
              CodeGen->EmitStmt("LATSampleIterMap[count] = access");
              CodeGen->tab_count--;
              CodeGen->EmitCode("} else { delete access; }");
            } else  {
              CodeGen->EmitStmt("LAT[addr] = count");
            }
          }
        }
        CodeGen->EmitStmt("count += 1");
//        if (EnableParallelOpt)
//          CodeGen->EmitStmt("m.unlock()");
      } else {
        // bypass this accses otherwise
        CodeGen->EmitComment(RefNode->getRefExprString()
                             + " will not access " + TargetRef->getArrayNameString());
        CodeGen->EmitStmt("count += 1");
        if (InterleavingTechnique == 0) {
          if (QueryEngine->GetParallelLoopDominator(RefNode))
            CodeGen->EmitStmt("parallel_count += 1");
          else
            CodeGen->EmitStmt("sequential_count += 1");
        }
      }
      CodeGen->tab_count--;
      CodeGen->EmitCode("} // end of if (start_reuse_search)");
      if (branchIter+1 != Branch->neighbors.end()) {
        SPSTNode *NextNode = *(branchIter+1);
        codess << "progress[tid_to_run]->increment(\""
               << SPSNodeNameTable[NextNode] << "\")";
      } else if (MergePoint) {
        codess << "progress[tid_to_run]->increment(\""
               << SPSNodeNameTable[MergePoint] << "\")";
      }
      CodeGen->EmitStmt(codess.str());
      codess.str("");
      if (type == UNIFORM_INTERLEAVING) {
        CodeGen->EmitStmt("worker_thread_iterator++");
      }
      CodeGen->EmitStmt("continue");
      CodeGen->tab_count--;
      CodeGen->EmitCode("} /* end of check to "
                        + SPSNodeNameTable[RefNode] + " */");
    } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(*branchIter)) {
      if (!Branch->neighbors[0]->neighbors.empty())
        PerRefParallelSamplerBranchBodyGenImpl(dynamic_cast<DummyTNode *>(Branch->neighbors[0]),
                                               type, MergePoint, TargetRef);
      if (!Branch->neighbors[1]->neighbors.empty())
        PerRefParallelSamplerBranchBodyGenImpl(dynamic_cast<DummyTNode *>(Branch->neighbors[1]),
                                               type, MergePoint, TargetRef);
    }
    branchIter++;
  }
}


// Note: it is possible that Child has both LOWER_BOUND and UPPER_BOUND
// dependency
//
pair<LoopTNode *, LoopTNode *>
ModelCodeGenProWrapperPass::CheckDependenciesInPath(LoopTNode *Child, AccPath &Path)
{
  IVDepNode *ParentNode = nullptr;
  pair<LoopTNode *, LoopTNode *> result = make_pair(nullptr, nullptr);

  auto path_riter = Path.rbegin();
  while (path_riter != Path.rend()) {
    LoopTNode *LoopInPath = dynamic_cast<LoopTNode *>(*path_riter);
    assert(LoopInPath && "All element in path should be in LoopTNode type");
    LLVM_DEBUG(dbgs() << "Check " << LoopInPath->getLoopStringExpr() << "\n");
    if (Child == LoopInPath)
      break;
    for (auto ivdep : getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
        .InductionVarDepForest) {
      if (ivdep->isNodeOf(LoopInPath->getInductionPhi())) {
        ParentNode = ivdep;
        LLVM_DEBUG(dbgs() << "ParentNode Found for " << LoopInPath->getLoopStringExpr() << "\n");
        break;
      }
    }
    if (!ParentNode) {
      path_riter++;
      continue;
    }
    DIRECTION deptype = ParentNode->isParentOf(Child->getInductionPhi());
    if (deptype == LOWER_BOUND_DEP) {
      LLVM_DEBUG(dbgs() << Child->getLoopStringExpr() << " has LOWER_BOUND_DEP on " << LoopInPath->getLoopStringExpr() << "\n");
      result.first = LoopInPath;
      // need reset the ParentNode for the UPPER_BOUND_DEP analysis.
      // When search for UPPER_BOUND_DEP later, if we do not clear the ParentNode,
      // the dependence checker can report wrong result if there is no ParentNode
      // found then updated when traversing later loops
      ParentNode = nullptr;
    }
    else if (deptype == UPPER_BOUND_DEP) {
      LLVM_DEBUG(dbgs() << Child->getLoopStringExpr() << " has UPPER_BOUND_DEP on " << LoopInPath->getLoopStringExpr() << "\n");
      result.second = LoopInPath;
      // need reset the ParentNode for the LOWER_BOUND_DEP analysis.
      // When search for LOWER_BOUND_DEP later, if we do not clear the ParentNode,
      // the dependence checker can report wrong result if there is no ParentNode
      // found then updated when traversing later loops
      ParentNode = nullptr;
    } else if (deptype == DUAL_DEP) {
      LLVM_DEBUG(dbgs() << Child->getLoopStringExpr() << " has LOWER_BOUND_DEP and UPPER_BOUND_DEP on "
                        << LoopInPath->getLoopStringExpr() << "\n");
      result.first = LoopInPath;
      result.second = LoopInPath;
    }
    if (result.first && result.second) // if both bound are found, no need to find others
      break;
    path_riter++;
  }
  return result;
}

bool ModelCodeGenProWrapperPass::hasInductionVarDependenceChildren(LoopTNode *LoopNode)
{
  IVDepNode *ParentNode = nullptr;
  for (auto ivdep : getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
      .InductionVarDepForest) {
    if (ivdep->isNodeOf(LoopNode->getInductionPhi())) {
      ParentNode = ivdep;
      break;
    }
  }
  if (!ParentNode)
    return false;
  return !ParentNode->dependences.empty();
}

bool ModelCodeGenProWrapperPass::hasParallelLoopInductionVar(Value *V)
{
  bool retVal = false;
  if (!V) {
    return false;
  }
  if (isa<ConstantInt>(V)) {
    return false;
  } else if (isa<ConstantFP>(V)) {
    return false;
  } else if (isa<Argument>(V)) {
    Argument *Arg = dyn_cast<Argument>(V);
    return false;
  }
  Instruction *I = dyn_cast<Instruction>(V);
  switch (I->getOpcode()) {
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
    for (unsigned i = 0; i < I->getNumOperands(); i++) {
      retVal |= hasParallelLoopInductionVar(I->getOperand(i));
    }
    break;
  }
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::Load: {
    retVal |= hasParallelLoopInductionVar(I->getOperand(0));
    break;
  }
  case Instruction::PHI: {
    PHINode *Phi = dyn_cast<PHINode>(I);
    if (Phi->getNumIncomingValues() == 1) {
      // in lcssa form, the array subscript could also be represented by phi
      // node with only one branches, in this case, we have to pass its operand to ValueToStringExpr() for example: i64 %idxprom33, i64 %idxprom35 %idxprom33.lcssa = phi i64 [ %idxprom33, %for.cond29 ]
      return hasParallelLoopInductionVar(Phi->getOperand(0));
    }
    for (auto loop : LoopNodes) {
      if (loop->isParallelLoop() && loop->getInductionPhi() == Phi) {
        retVal = true;
        break;
      }
    }
    break;
  }
  default:
    break;
  }
  return retVal;
}

bool ModelCodeGenProWrapperPass::isInParallelLoop(RefTNode *RefNode)
{
  AccPath path;
  QueryEngine->GetPath(RefNode, nullptr, path);
  auto pathRIter = path.rbegin();
  while (pathRIter != path.rend()) {
    LoopTNode *Loop = dynamic_cast<LoopTNode *>(*pathRIter);
    assert(Loop && "Items inside path should be an instance of LoopTNode");
    if (Loop->isParallelLoop())
      return true;
    pathRIter++;
  }
  return false;
}

/// Call for every function
bool ModelCodeGenProWrapperPass::runOnFunction(Function &F)
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
  SEWP = &getAnalysis<ScalarEvolutionWrapperPass>();
  BAWP = &getAnalysis<BranchAnalysis::BranchAnalysisWrapperPass>();

  /* previous analysis result */
  InductionVarNameTable =
      getAnalysis<InductionVarAnalysis::InductionVarAnalysis>()
          .InductionVarNameTable;

  ParentMappingInAbstractionTree =
      getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>()
          .ImmediateParentMapping;
  LoopNodes =
      getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>().LoopNodeList;
  PointerAliasPairs =
      getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>()
          .PointerAliasCandidate;
  SPSNodeNameTable = getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>()
      .NodeToRefNameMapping;
  Graphs = getAnalysis<AccessGraphAnalysis::AccessGraphAnalysisPass>()
      .TopNodeToGraphMapping;

  ModelValidateLoops = getAnalysis<ModelAnalysis::ModelValidationAnalysis>().ModelApplicableLoops;

  for (auto mapping: SPSNodeNameTable) {
    RefNameToNodeMapping[mapping.second] = mapping.first;
  }

  /* utils */
  Translator = new StringTranslator(&SEWP->getSE(), InductionVarNameTable);
  QueryEngine = &getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>()
      .getQueryEngine();
  CodeGen = new SamplerCodeGenerator();
  Analyzer = new SampleNumberAnalyzer(
      getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>().TreeRoot,
      SamplingRatio);

  /* parse command line argument */
  switch (SamplingTechnique) {
  case 1:
    SamplingMethod = SEQUENTIAL_START;
    break;
  case 2:
    SamplingMethod = RANDOM_START;
    break;
  case 3:
    SamplingMethod = BURSTY;
    break;
  default:
    SamplingMethod = NO_SAMPLE;
    break;
  }
  EnableSampling = (SamplingMethod != NO_SAMPLE) && (SamplingRatio < 100);
  // If EnableModelOpt option is given, the model will be applied if the parallel
  // loop
  // 1) can be parallelized by the OpenMP static scheduling and
  // 2) threads are interleaved uniformly
  // Here we check the condition 2)
  if (EnableSampling) {
    // compute sample number
    Analyzer->analyze();

    for (auto node : SPSNodeNameTable) {
      if (RefTNode *ref = dynamic_cast<RefTNode *>(node.first)) {
        AccPath path;
        QueryEngine->GetPath(ref, nullptr, path);
        if(path.empty()) {
          LLVM_DEBUG(dbgs() << "SKIP the sample number computation for "
                            << SPSNodeNameTable[ref] << "\n");
          continue;
        }
        unsigned sample_number = Analyzer->getSampleNumberForRef(ref);
        PerReferenceSampleCnt[ref] = sample_number;
        LLVM_DEBUG(dbgs() << ref->getRefExprString() << " (" << node.second
                          << ") \t" << sample_number << "\n");
      }
    }
  }

  /* codegen start */
  HeaderGen();
  RefAddressFuncGen();
  ModelUtilGen();
  SamplerBodyGen(true);
  MainFuncGen();
  return false;
}

void ModelCodeGenProWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
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
  AU.addRequired<AccessGraphAnalysis::AccessGraphAnalysisPass>();
  AU.addPreserved<AccessGraphAnalysis::AccessGraphAnalysisPass>();
  AU.addRequired<ModelAnalysis::ModelValidationAnalysis>();
  AU.addPreserved<ModelAnalysis::ModelValidationAnalysis>();
  AU.addRequired<ReferenceAnalysis::ReferenceAnalysis>();
  AU.addPreserved<ReferenceAnalysis::ReferenceAnalysis>();
}

} // end of namespace ModelCodeGen