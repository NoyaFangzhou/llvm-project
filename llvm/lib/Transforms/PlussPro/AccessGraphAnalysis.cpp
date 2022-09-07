//
// Created by noya-fangzhou on 11/13/21.
//

#include "AccessGraphAnalysis.h"
#include "BranchAnalysis.h"
#include "InductionVarAnalysis.h"
#include "LoopAnalysisWrapperPass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils.h"
#include <sstream>
#define DEBUG_TYPE	"accgraph-analysis"

using namespace std;
using namespace llvm;

//extern cl::opt<string> FunctionName;

cl::opt<bool> ViewAccessGraph(
    "view-graph", cl::init(false), cl::NotHidden,
    cl::desc("Output the access graph in DOT format"));

STATISTIC(AccGraphCount, "Number of Access Graph generated");
STATISTIC(NumNodeInsideGraph, "Number of Vertices inside all Graphs");
STATISTIC(NumEdgeInsideGraph, "Number of Edges inside all Graphs");

namespace AccessGraphAnalysis {

char AccessGraphAnalysisPass::ID = 0;
static RegisterPass<AccessGraphAnalysisPass>
    X("accgraph-analysis", "Pass to convert the loop abstraction"
                           "tree to the access graph");

AccessGraphAnalysisPass::AccessGraphAnalysisPass() : FunctionPass(ID) {}

void AccessGraphAnalysisPass::BuildAccessGraph(SPSTNode *Root)
{
  if (dynamic_cast<LoopTNode *>(Root) || dynamic_cast<BranchTNode *>(Root)) {
    AccGraph *G = new AccGraph(Root);
    TopNodeToGraphMapping[Root] = G;
    AccGraphCount++;
    NumNodeInsideGraph += G->GetNumNodes();
    NumEdgeInsideGraph += G->GetNumEdges();
  }
}

void AccessGraphAnalysisPass::ViewGraph(AccGraph *G, string title)
{
  // we first declare all nodes
  set<SPSTNode *> Nodes = G->GetAllNodesInGraph();
  string tab = "\t";
  string code = "digraph \"Access Graph " + title + "\" {\n";
  code += (tab + "label=\"Access Graph;\"\n");
  auto node_iter = Nodes.begin();
  stringstream ss;
  string node_addr = "", fillcolor = "", shape = "oval", content="";
  while (node_iter != Nodes.end()) {
    ss.str("");
    ss << (*node_iter);
    node_addr = ss.str();
    if (RefTNode *Ref = dynamic_cast<RefTNode *>(*node_iter)) {
      content = Ref->getRefExprString();
    } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(*node_iter)) {
      content = Branch->getConditionExpr();
    }
    code += (tab + "Node" + node_addr + " " + "[fillcolor=\"" + fillcolor + "\","
             + "style=\"rounded,filled\",shape=" + shape + ","
             + "label=\"{AccNode:\\l  " + content + "\\l}\"];\n");
    SmallVector<AccGraphEdge *, 16> Edges;
    G->GetEdgesWithSource(*node_iter, Edges);
    if (!Edges.empty()) {
      string sink_addr = "", edge_label = "";
      for (auto edge : Edges) {
        edge_label = "";
        ss.str("");
        ss << edge->getSink();
        sink_addr = ss.str();
        LoopTNode *L = edge->getCarryLoop();
        if (L) {
          edge_label = "[ label=\"" + L->getLoopStringExpr() + "\" ]";
        }
        code += tab + "Node" + node_addr + " -> Node" + sink_addr
                + " " + edge_label + ";\n";
      }
    }
    node_iter++;
  }
  // next we connect these nodes using edges


  code += "}\n";
  LLVM_DEBUG(dbgs() << code << "\n");

}

/* Analysis pass main function */
bool AccessGraphAnalysisPass::runOnFunction(Function &F)
{
  if (!FunctionName.empty() && F.getName().str().find(FunctionName) == string::npos) {
    // the function name is given, only the given function will be analyzed
    LLVM_DEBUG(dbgs() << F.getName() << " will be skipped\n");
    return false;
  }
  TreeRoot = getAnalysis<PlussLoopAnalysis::LoopAnalysisWrapperPass>().TreeRoot;
  if (TreeRoot) {
#if defined(__APPLE__) && defined(__MACH__)
    LLVM_DEBUG(dbgs() << __FILE_NAME__ << " on " << F.getName() << "\n");
#elif defined(__linux__)
    LLVM_DEBUG(dbgs() << __FILE__ << " on " << F.getName() << "\n");
#endif
    queue<SPSTNode *> queue;
    queue.push(TreeRoot);
    while (!queue.empty()) {
      SPSTNode *Node = queue.front();
      queue.pop();
      if (LoopTNode *LN = dynamic_cast<LoopTNode *>(Node)) {
        if (LN->isParallelLoop()) {
          BuildAccessGraph(LN);
        }
      }
      auto neighbor_iter = Node->neighbors.begin();
      for (; neighbor_iter != Node->neighbors.end(); ++neighbor_iter) {
        // only branch node, dummy node (one branch condition) and loop node
        // will be examined.
        if (!dynamic_cast<RefTNode *>(*neighbor_iter)) {
          queue.push(*neighbor_iter);
        }
      }
    }
  }
  if (ViewAccessGraph) {
    if (TreeRoot && !TopNodeToGraphMapping.empty()) {
      LLVM_DEBUG(dbgs() << "Graph: \n");
      unsigned i = 0;
      for (auto p : TopNodeToGraphMapping) {
        ViewGraph(p.second, to_string(i));
        i++;
      }
    } else {
      LLVM_DEBUG(dbgs() << "No available Access graph \n");
    }
  }
  return false;
}

void AccessGraphAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const
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
  AU.addRequired<PlussLoopAnalysis::LoopAnalysisWrapperPass>();
}


} // end of namespace AccessGraphAnalysis