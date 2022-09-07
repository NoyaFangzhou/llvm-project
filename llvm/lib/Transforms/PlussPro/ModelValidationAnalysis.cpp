//
// Created by noya-fangzhou on 1/6/22.
//
#include "llvm/Transforms/Utils.h"
#include "llvm/ADT/Statistic.h"
#include "ModelValidationAnalysis.h"
#include "LoopAnalysisWrapperPass.h"
#include "PlussAbstractionTreeAnalysis.h"
#include "InductionVarAnalysis.h"
#define DEBUG_TYPE  "model-analysis"

STATISTIC(NumModelApplicableLoop, "Number of loop nests that are applicable "
                                  "by the model");
STATISTIC(NumLoopNests, "Number of loop nests");

STATISTIC(NumParallelLoopNests, "Number of parallel loop nests");

namespace ModelAnalysis {

char ModelValidationAnalysis::ID = 0;
static RegisterPass<ModelValidationAnalysis>
    X("model-analysis", "Pass that check the number of loop nests "
                        "that model can be applied");

ModelValidationAnalysis::ModelValidationAnalysis() : FunctionPass(ID) {}

void ModelValidationAnalysis::analysis()
{
  // check number of loop regions that can be applicable by models
  auto treeIter = TreeRoot->neighbors.begin();
  while (treeIter != TreeRoot->neighbors.end()) {
    if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(*treeIter)) {
      NumLoopNests++;
    }
    analysisImpl(*treeIter, *QueryEngine);
    treeIter++;
  }

}

void ModelValidationAnalysis::analysisImpl(SPSTNode *Node,
                                           TreeStructureQueryEngine &QueryEngine)
{
  if (LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(Node)) {
    // Loop is parallel, no induction dependence and no branches inside
    if (LoopNode->isParallelLoop()) {
      NumParallelLoopNests++;
      if (QueryEngine.hasConstantLoopBound(LoopNode)
          && !QueryEngine.hasBranchInside(LoopNode)) {
        // if all its child loop does not have induction variable dependencies
        // (works with OpenMP static scheduling)
        if (!hasInductionVarDependenceChildren(LoopNode)) {
          NumModelApplicableLoop++;
          ModelApplicableLoops.insert(LoopNode);
        }
      }
    } else {
      auto neighbor_iter =  LoopNode->neighbors.begin();
      for (; neighbor_iter != LoopNode->neighbors.end(); ++neighbor_iter)
        analysisImpl(*neighbor_iter, QueryEngine);
    }
  } else if (RefTNode *RefNode = dynamic_cast<RefTNode *>(Node)) {

  } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(Node)) {
    if (!Branch->neighbors[0]->neighbors.empty()) {
      auto neighbor_iter = Branch->neighbors[0]->neighbors.begin();
      for (; neighbor_iter != Branch->neighbors[0]->neighbors.end(); ++neighbor_iter)
        analysisImpl(*neighbor_iter, QueryEngine);
    }
    if (!Branch->neighbors[1]->neighbors.empty()) {
      auto neighbor_iter = Branch->neighbors[1]->neighbors.begin();
      for (; neighbor_iter != Branch->neighbors[1]->neighbors.end(); ++neighbor_iter)
        analysisImpl(*neighbor_iter, QueryEngine);
    }
  } else if (DummyTNode *Dummy = dynamic_cast<DummyTNode *>(Node)) {
    auto neighbor_iter =  Dummy->neighbors.begin();
    for (; neighbor_iter != Dummy->neighbors.end(); ++neighbor_iter)
      analysisImpl(*neighbor_iter, QueryEngine);
  }
}

bool ModelValidationAnalysis::hasInductionVarDependenceChildren(LoopTNode *LoopNode)
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

unsigned ModelValidationAnalysis::CountPotentialReuseReferences(LoopTNode *LoopNode)
{
  unsigned count = 0;
  DenseMap<SPSTNode *, string> SPSNodeNameTable = getAnalysis<TreeAnalysis::PlussAbstractionTreeAnalysis>()
      .NodeToRefNameMapping;
  vector<RefTNode *> ReferencesInParallelLoop;
  for (auto entry : SPSNodeNameTable) {
    if (RefTNode *RefNode = dynamic_cast<RefTNode *>(entry.first)) {
      if (QueryEngine->GetParallelLoopDominator(RefNode) == LoopNode)
        ReferencesInParallelLoop.push_back(RefNode);
    }
  }

  unsigned i = 0, j = 0;
  for (i = 0; i < ReferencesInParallelLoop.size(); i++) {
    RefTNode *RefA = ReferencesInParallelLoop[i];
    for (j = i+1; j < ReferencesInParallelLoop.size(); j++) {
      RefTNode *RefB = ReferencesInParallelLoop[j];
      if (QueryEngine->areAccessToSameArray(RefA, RefB))
        count++;
    }
  }
  return count;
}

/// Call for every function
bool ModelValidationAnalysis::runOnFunction(Function &F)
{
  if (!FunctionName.empty() && F.getName().str().find(FunctionName) == string::npos) {
    // the function name is given, only the given function will be analyzed
    LLVM_DEBUG(dbgs() << F.getName() << " will be skipped\n");
    return false;
  }
  TreeRoot =
      getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>().TreeRoot;
  QueryEngine = new TreeStructureQueryEngine(TreeRoot);
#if defined(__APPLE__) && defined(__MACH__)
  LLVM_DEBUG(dbgs() << __FILE_NAME__ << " on " << F.getName() << "\n");
#elif defined(__linux__)
  LLVM_DEBUG(dbgs() << __FILE__ << " on " << F.getName() << "\n");
#endif
  analysis();
  // after alias analysis, we now merge those
  return false;
}

void ModelValidationAnalysis::getAnalysisUsage(AnalysisUsage &AU) const
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
  AU.addRequired<TreeAnalysis::PlussAbstractionTreeAnalysis>();
  AU.addPreserved<TreeAnalysis::PlussAbstractionTreeAnalysis>();
}

} // end of ModelAnalysis namespace
