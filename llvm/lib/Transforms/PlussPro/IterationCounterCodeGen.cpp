//
// Created by noya-fangzhou on 1/21/22.
//

#include "IterationCounterCodeGen.h"
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
#include "llvm/Transforms/Utils.h"
#include <algorithm>

#define DEBUG_TYPE "access-counter-codegen"

using namespace std;
using namespace llvm;

namespace IterationCounterCodeGen {

char IterationCountCodeGenWrapperPass::ID = 0;
static RegisterPass<IterationCountCodeGenWrapperPass>
    X("counter-codegen", "Pass that generate the access counter code");

IterationCountCodeGenWrapperPass::IterationCountCodeGenWrapperPass() : FunctionPass(ID) {}

string IterationCountCodeGenWrapperPass::EmitBranchCondExpr(BranchTNode *Branch)
{
  TranslateStatus status = SUCCESS;
  string cond = "true";
  stringstream  ss;
//  Translator->ValueToStringExpr(Branch->getCondition(), status);
  CmpInst *CI = dyn_cast<CmpInst>(Branch->getCondition());
  string tmp = "";
  if (Branch->neighbors[0]->neighbors.empty()) {
    tmp = (Translator->ValueToStringExpr(CI->getOperand(0), status)
           + Translator->PredicateToStringExpr(CI->getInversePredicate(), status)
           + Translator->ValueToStringExpr(CI->getOperand(0), status));
  } else {
    tmp = (Translator->ValueToStringExpr(CI->getOperand(0), status)
           + Translator->PredicateToStringExpr(CI->getPredicate(), status)
           + Translator->ValueToStringExpr(CI->getOperand(0), status));
  }
  if (status == SUCCESS) {
    cond = tmp;
  }
  ss << "if (" << cond << ")";
  return ss.str();
}


string IterationCountCodeGenWrapperPass::EmitLoopNodeExpr(LoopTNode *LoopNode,
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
    if (isSampledLoop) {
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

void IterationCountCodeGenWrapperPass::HeaderGen()
{
  CodeGen->EmitCode("#include <iostream>");
  CodeGen->EmitCode("#include <cmath>");
  CodeGen->EmitCode("#include <random>");
  CodeGen->EmitCode("#include <algorithm>");
  CodeGen->EmitCode("#include \"pluss.h\"");
  CodeGen->EmitStmt("using namespace std");
  CodeGen->EmitStmt("unsigned sequential_count = 0");
  CodeGen->EmitStmt("unsigned parallel_count = 0");
}


void IterationCountCodeGenWrapperPass::MainFuncGen()
{
  CodeGen->EmitCode("int main() {");
  CodeGen->tab_count++;
  CodeGen->EmitFunctionCall("counter", {});
  CodeGen->EmitStmt("cout << \"sequential accesses: \" << sequential_count << endl");
  CodeGen->EmitStmt("cout << \"parallel accesses: \" << parallel_count << endl");
  CodeGen->EmitStmt("cout << \"ratio: \" << (double)parallel_count / (parallel_count + sequential_count) << endl");
  CodeGen->EmitStmt("return 0");
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
  return;
}

void IterationCountCodeGenWrapperPass::CounterBodyGen()
{
  CodeGen->EmitCode("void counter() {");
  CodeGen->tab_count++;
  stringstream ss;
  auto topiter = getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
      .TreeRoot->neighbors.begin();
  unordered_set<string> arrays;
  for (auto entry :SPSNodeNameTable) {
    if (RefTNode *RefNode = dynamic_cast<RefTNode *>(entry.first)) {
      if (arrays.find(RefNode->getArrayNameString()) == arrays.end()) {
        CodeGen->EmitStmt("unordered_map<unsigned, unsigned long> LAT_" +
                          RefNode->getArrayNameString());
        arrays.insert(RefNode->getArrayNameString());
      }
    }
  }
  for (; topiter != getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
      .TreeRoot->neighbors.end();
         topiter++) {
    CounterBodyGenImpl(*topiter);
  }
  CodeGen->tab_count--;
  CodeGen->EmitCode("}");
}

void IterationCountCodeGenWrapperPass::CounterBodyGenImpl(SPSTNode *Root)
{
  if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(Root)) {
    string forloop = EmitLoopNodeExpr(LoopNode);
    CodeGen->EmitCode(forloop + " {");
    CodeGen->tab_count++;
    for (auto neighbor : LoopNode->neighbors) {
      CounterBodyGenImpl(neighbor);
    }
    CodeGen->tab_count--;
    CodeGen->EmitCode("}");
  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Root)) {
    string branch = EmitBranchCondExpr(Branch);
    // generate the code if condition is true
    if (!Branch->neighbors[0]->neighbors.empty()) {
      CodeGen->EmitCode(branch + " {");
      CodeGen->tab_count++;
      CounterBodyGenImpl(Branch->neighbors[0]);
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
      CounterBodyGenImpl(Branch->neighbors[1]);
      CodeGen->tab_count--;
    }
    CodeGen->EmitCode("}");
  } else if (RefTNode *RefNode = dynamic_cast<RefTNode *>(Root)) {
    stringstream ss;
    ss << "Generate address check func call for " << RefNode->getRefExprString();
    CodeGen->EmitComment(ss.str());
    ss.str("");
    if (QueryEngine->GetParallelLoopDominator(RefNode)) {
      CodeGen->EmitStmt("parallel_count += 1");
    } else {
      CodeGen->EmitStmt("sequential_count += 1");
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Root)) {
    for (auto neighbor : Dummy->neighbors) {
      CounterBodyGenImpl(neighbor);
    }
  }
}

/// Call for every function
bool IterationCountCodeGenWrapperPass::runOnFunction(Function &F)
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

  /* utils */
  Translator = new StringTranslator(&SEWP->getSE(), InductionVarNameTable);
  QueryEngine = &getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>()
      .getQueryEngine();
  CodeGen = new SamplerCodeGenerator();

  /* codegen start */
  HeaderGen();
  CounterBodyGen();
  MainFuncGen();
  return false;
}

void IterationCountCodeGenWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
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
}

} // end of namespace
