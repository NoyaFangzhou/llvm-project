//
// Created by noya-fangzhou on 11/17/21.
//

#include "llvm/Transforms/Utils.h"
#include "llvm/ADT/Statistic.h"
#include "PlussAbstractionTreeAnalysis.h"
#include "InductionVarAnalysis.h"
#include "LoopAnalysisWrapperPass.h"
#define DEBUG_TYPE  "tree-analysis"

STATISTIC(NumPointerAlias, "Number of references that are viewed as pointer"
                           "alias");

namespace TreeAnalysis {

char PlussAbstractionTreeAnalysis::ID = 0;
static RegisterPass<PlussAbstractionTreeAnalysis>
    X("tree-analysis", "Pass that collect metadata from the abstraction tree"
                       ", which will be used in later analysis");

PlussAbstractionTreeAnalysis::PlussAbstractionTreeAnalysis() : FunctionPass(ID) {}

void PlussAbstractionTreeAnalysis::BuildParentMapping(SPSTNode *Root)
{
  if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Root)) {
    auto neighbor_iter = Dummy->neighbors.begin();
    for (; neighbor_iter != Dummy->neighbors.end(); neighbor_iter++) {
      BuildParentMapping(*neighbor_iter);
    }
  } else if (LoopTNode *LoopNode = dynamic_cast<LoopTNode*>(Root)) {
    LoopNodeList.push_back(LoopNode);
    auto neighbor_iter = LoopNode->neighbors.begin();
    for (; neighbor_iter != LoopNode->neighbors.end(); neighbor_iter++) {
      if (dynamic_cast<LoopTNode*>(*neighbor_iter) || dynamic_cast<RefTNode *>(*neighbor_iter)
          || dynamic_cast<BranchTNode *>(*neighbor_iter)) {
        ImmediateParentMapping[*neighbor_iter] = LoopNode;
        BuildParentMapping(*neighbor_iter);
      }
    }
  } else if (RefTNode *RefNode = dynamic_cast<RefTNode*>(Root)) {
    RefNodeOrderedList.push_back(RefNode);
    return;
  } else if (BranchTNode *BranchNode = dynamic_cast<BranchTNode*>(Root)) {
    // true branch
    if (!BranchNode->neighbors[0]->neighbors.empty()) {
      auto neighbor_iter = BranchNode->neighbors[0]->neighbors.begin();
      for (; neighbor_iter != BranchNode->neighbors[0]->neighbors.end();
           neighbor_iter++) {
        if (ImmediateParentMapping.find(BranchNode) !=
            ImmediateParentMapping.end()) {
          ImmediateParentMapping[*neighbor_iter] =
              ImmediateParentMapping[BranchNode];
        }
        BuildParentMapping(*neighbor_iter);
      }
    }
    // false branch
    if (!BranchNode->neighbors[1]->neighbors.empty()) {
      auto neighbor_iter = BranchNode->neighbors[1]->neighbors.begin();
      for (; neighbor_iter != BranchNode->neighbors[1]->neighbors.end(); neighbor_iter++) {
        if (ImmediateParentMapping.find(BranchNode) != ImmediateParentMapping.end()) {
          ImmediateParentMapping[*neighbor_iter] = ImmediateParentMapping[BranchNode];
        }
        BuildParentMapping(*neighbor_iter);
      }
    }
  }
}

void PlussAbstractionTreeAnalysis::BuildPointerAliasMap()
{
  DenseMap<RefTNode *, RefTNode *> AliasPair;
  for (auto RefNode : RefNodeOrderedList) {
    auto ref_iter = RefNodeOrderedList.begin();
    while (ref_iter != RefNodeOrderedList.end()) {
      // ignore the case where the two references are the same
      // or they access the same array base
      if (*ref_iter == RefNode || (*ref_iter)->getBase() == RefNode->getBase()) {
        ref_iter++;
        continue;
      }
      // query RefNode and (*ref_iter) 's instruction to SVF/AliasAnalysis
      AliasResult result = AA->alias(RefNode->getBase(), (*ref_iter)->getBase());
      if (result == MustAlias) {
        NumPointerAlias++;
        AliasPair.insert(make_pair(RefNode, *ref_iter));
      }
      ref_iter++;
    }
  }

  // now we give those references that alias a new common name
  unsigned AliasCount = 0;
  for (auto pair : AliasPair) {
    if (PointerAliasCandidate.find(pair.first) == PointerAliasCandidate.end()) {
      string name = "alias" + to_string(AliasCount);
      PointerAliasCandidate.insert(make_pair(pair.first, name));
      PointerAliasCandidate.insert(make_pair(pair.second, name));
      AliasCount++;
    }
  }

}

void PlussAbstractionTreeAnalysis::GenerateReferenceName()
{
  // init
  BranchCount = 0;
  for (auto RefNode : RefNodeOrderedList) {
    if (PerArrayReferenceCount.find(RefNode->getBase())
        == PerArrayReferenceCount.end()) {
      PerArrayReferenceCount[RefNode->getBase()] = 0;
    }
  }
  GenerateReferenceNameImpl(TreeRoot);
}

void PlussAbstractionTreeAnalysis::GenerateReferenceNameImpl(SPSTNode *Root)
{
  if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Root)) {
    auto neighbor_iter = Dummy->neighbors.begin();
    for (; neighbor_iter != Dummy->neighbors.end(); neighbor_iter++) {
      GenerateReferenceNameImpl(*neighbor_iter);
    }
  } else if (LoopTNode *LoopNode = dynamic_cast<LoopTNode*>(Root)) {
    auto neighbor_iter = LoopNode->neighbors.begin();
    for (; neighbor_iter != LoopNode->neighbors.end(); neighbor_iter++) {
      GenerateReferenceNameImpl(*neighbor_iter);
    }
  } else if (RefTNode *RefNode = dynamic_cast<RefTNode*>(Root)) {
    assert(PerArrayReferenceCount.find(RefNode->getBase()) != PerArrayReferenceCount.end()
           && "All RefTNode should have a mapping in PerArrayReferenceCount");
    TranslateStatus status = SUCCESS;
    stringstream idss;
    idss << Translator->ValueToStringExpr(RefNode->getBase(), status) << PerArrayReferenceCount[RefNode->getBase()];
    NodeToRefNameMapping.insert(make_pair(RefNode, idss.str()));
    idss.str("");
    PerArrayReferenceCount[RefNode->getBase()]++;
  } else if (BranchTNode *BranchNode = dynamic_cast<BranchTNode*>(Root)) {
    stringstream idss;
    idss << "branch" << BranchCount;
    NodeToRefNameMapping.insert(make_pair(BranchNode, idss.str()));
    BranchCount++;
    idss.str("");
    // true branch
    if (!BranchNode->neighbors[0]->neighbors.empty()) {
      auto neighbor_iter = BranchNode->neighbors[0]->neighbors.begin();
      for (; neighbor_iter != BranchNode->neighbors[0]->neighbors.end();
           neighbor_iter++) {
        GenerateReferenceNameImpl(*neighbor_iter);
      }
    }
    // false branch
    if (!BranchNode->neighbors[1]->neighbors.empty()) {
      auto neighbor_iter = BranchNode->neighbors[1]->neighbors.begin();
      for (; neighbor_iter != BranchNode->neighbors[1]->neighbors.end(); neighbor_iter++) {
        GenerateReferenceNameImpl(*neighbor_iter);
      }
    }
  }
}

void PlussAbstractionTreeAnalysis::DoReferenceReachableAnalysis()
{
  LLVM_DEBUG(dbgs() << "Reachable analysis: \n");
  for (auto ref : RefNodeOrderedList) {
    LLVM_DEBUG(dbgs() << "To find reuse of reference " << NodeToRefNameMapping[ref]
                      << ":\n");
    for (auto check: RefNodeOrderedList) {
      if (ref == check)
        continue;
      if (QueryEngine->isReachable(ref, check))
        LLVM_DEBUG(dbgs() << "\t ---- " << NodeToRefNameMapping[check] << "\n");
    }
    LLVM_DEBUG(dbgs() << "\n");
  }
}

void PlussAbstractionTreeAnalysis::FindInductionVarBound()
{
  unordered_map<PHINode *, string> InductionVarName =
      getAnalysis<InductionVarAnalysis::InductionVarAnalysis>().InductionVarNameTable;
  unordered_map<PHINode *, string> InductionVarBoundTable;
  vector<LoopTNode *> UnParsableInductionVar;
  string tmp = "";
  TranslateStatus status = SUCCESS;
  for (auto loop : LoopNodeList) {
    LLVM_DEBUG(dbgs() << loop->getLoopStringExpr() << "\n");
    string induction_name = InductionVarName[loop->getInductionPhi()];
    transform(induction_name.begin(), induction_name.end(),
              induction_name.begin(), ::toupper);
    string bound = "N" + induction_name;
    if (QueryEngine->hasConstantLoopBound(loop)) {
      tmp = Translator->ValueToStringExpr(loop->getLoopBound()->FinalValue,
                                          status);

      if (status == SUCCESS)
        bound = tmp;
      InductionVarBoundTable[loop->getInductionPhi()] = bound;
    } else if (QueryEngine->hasConstantLoopUpperBound(loop)) {
      tmp = Translator->ValueToStringExpr(loop->getLoopBound()->FinalValue,
                                          status);
      if (status == SUCCESS)
        bound = tmp;
      InductionVarBoundTable[loop->getInductionPhi()] = bound;
    } else {
      LLVM_DEBUG(dbgs() << "Check Induction Variable Dependency for "
                        << loop->getLoopStringExpr() << "\n");
      IVDepNode *ChildNode = nullptr;
      for (auto ivdep :
           getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>()
               .InductionVarDepForest) {
        LLVM_DEBUG(dbgs() << "Visit IVDepNode for " << InductionVarName[ivdep->getInduction()] << "\n");
        if (ivdep->isParentOf(loop->getInductionPhi()) == DUAL_DEP
            || ivdep->isParentOf(loop->getInductionPhi()) == UPPER_BOUND_DEP) {
          if (InductionVarBoundTable.find(ivdep->getInduction()) !=
              InductionVarBoundTable.end()) {
            InductionVarBoundTable[loop->getInductionPhi()] =
                InductionVarBoundTable[ivdep->getInduction()];
            break;
          }
        }
      }
    }
  }
  LLVM_DEBUG(
      for (auto entry : InductionVarBoundTable) {
        dbgs() << InductionVarName[entry.first] << " -- " << entry.second << "\n";
      }
  );
  for (auto ref : RefNodeOrderedList) {
    if (ArrayBound.find(ref->getArrayNameString()) != ArrayBound.end()) {
      unsigned i = 0;
      for (; i < ref->getSubscripts().size(); i++) {
        Value *Index = ref->getSubscripts()[i];
        PHINode *InductionPhi = GetInductionVarFromArraySubscript(Index);
        if (InductionPhi) {
          if (ArrayBound[ref->getArrayNameString()][i] == "1") {
            ArrayBound[ref->getArrayNameString()][i] =
                InductionVarBoundTable[InductionPhi];
          } else if (ArrayBound[ref->getArrayNameString()][i] != InductionVarBoundTable[InductionPhi]) {
            bool isLHSConstant = isConstantString(ArrayBound[ref->getArrayNameString()][i]);
            bool isRHSConstant = isConstantString(InductionVarBoundTable[InductionPhi]);
            if (isLHSConstant && isRHSConstant) {
              ArrayBound[ref->getArrayNameString()][i] = to_string(
                  max(stoi(ArrayBound[ref->getArrayNameString()][i]),
                      stoi(InductionVarBoundTable[InductionPhi])));
            } else {
              ArrayBound[ref->getArrayNameString()][i] =
                  "max(" + ArrayBound[ref->getArrayNameString()][i] + "," +
                  InductionVarBoundTable[InductionPhi] + ")";
            }
          }
        }
      }
    } else {
      vector<string> index_bound;
      for (auto Index : ref->getSubscripts()) {
        PHINode *InductionPhi = GetInductionVarFromArraySubscript(Index);
        if (InductionPhi) {
          index_bound.push_back(InductionVarBoundTable[InductionPhi]);
        } else {
          index_bound.push_back("1");
        }
      }
      ArrayBound[ref->getArrayNameString()] = index_bound;
    }
  }
  LLVM_DEBUG(
      for (auto entry : ArrayBound) {
        dbgs() << entry.first;
        for (auto bound : entry.second) {
          dbgs() << "[" << bound << "]";
        }
        dbgs() << "\n";
      }
  );
}

PHINode * PlussAbstractionTreeAnalysis::GetInductionVarFromArraySubscript(Value *V)
{
  if (!V) {
    return nullptr;
  }
  if (isa<ConstantInt>(V)) {
    return nullptr;
  } else if (isa<ConstantFP>(V)) {
    return nullptr;
  } else if (isa<Argument>(V)) {
    Argument *Arg = dyn_cast<Argument>(V);
    return nullptr;
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
      PHINode* ret = GetInductionVarFromArraySubscript(I->getOperand(i));
      if (ret)
        return ret;
    }
    break;
  }
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::Load: {
    return GetInductionVarFromArraySubscript(I->getOperand(0));
    break;
  }
  case Instruction::PHI: {
    PHINode *Phi = dyn_cast<PHINode>(I);
    if (Phi->getNumIncomingValues() == 1) {
      // in lcssa form, the array subscript could also be represented by phi
      // node with only one branches, in this case, we have to pass its operand to ValueToStringExpr() for example: i64 %idxprom33, i64 %idxprom35 %idxprom33.lcssa = phi i64 [ %idxprom33, %for.cond29 ]
      return GetInductionVarFromArraySubscript(Phi->getOperand(0));
    }
    for (auto loop : LoopNodeList) {
      if (loop->getInductionPhi() == Phi) {
        return Phi;
      }
    }
    break;
  }
  default:
    break;
  }
}

TreeStructureQueryEngine &PlussAbstractionTreeAnalysis::getQueryEngine()
{
  return *QueryEngine;
}

/// Call for every function
bool PlussAbstractionTreeAnalysis::runOnFunction(Function &F)
{
  if (!FunctionName.empty() && F.getName().str().find(FunctionName) == string::npos) {
    // the function name is given, only the given function will be analyzed
    LLVM_DEBUG(dbgs() << F.getName() << " will be skipped\n");
    return false;
  }
  TreeRoot =
      getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>().TreeRoot;
#if defined(__APPLE__) && defined(__MACH__)
  LLVM_DEBUG(dbgs() << __FILE_NAME__ << " on " << F.getName() << "\n");
#elif defined(__linux__)
  LLVM_DEBUG(dbgs() << __FILE__ << " on " << F.getName() << "\n");
#endif
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  Translator = new StringTranslator(
      &getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
      getAnalysis<InductionVarAnalysis::InductionVarAnalysis>()
          .InductionVarNameTable);
  QueryEngine = new TreeStructureQueryEngine(TreeRoot);
  BuildParentMapping(TreeRoot);
  LLVM_DEBUG(dbgs() << ImmediateParentMapping.size()
                    << " Pairs in ParentMapping\n";
             dbgs() << RefNodeOrderedList.size() << " RefNodes in the tree\n");
  GenerateReferenceName();
  LLVM_DEBUG(
      // dump all node name
      for (auto mapping
           : NodeToRefNameMapping) {
        if (RefTNode *Ref = dynamic_cast<RefTNode *>(mapping.first)) {
          dbgs() << Ref->getRefExprString() << " " << mapping.second << "\n";
        } else if (BranchTNode *Branch =
                       dynamic_cast<BranchTNode *>(mapping.first)) {
          dbgs() << Branch->getConditionExpr() << " " << mapping.second
                 << "\n";
        }
      });
  BuildPointerAliasMap();
  DoReferenceReachableAnalysis();
  FindInductionVarBound();
  // after alias analysis, we now merge those
  return false;
}

void PlussAbstractionTreeAnalysis::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.setPreservesAll();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequiredID(LoopSimplifyID);
  AU.addPreservedID(LoopSimplifyID);
  AU.addRequiredID(LCSSAID);
  AU.addPreservedID(LCSSAID);
  AU.addRequired<InductionVarAnalysis::InductionVarAnalysis>();
  AU.addPreserved<InductionVarAnalysis::InductionVarAnalysis>();
  AU.addRequired<PlussLoopAnalysis::LoopAnalysisWrapperPass>();
  AU.addPreserved<PlussLoopAnalysis::LoopAnalysisWrapperPass>();
}


} // end of TreeAnalysis namespace