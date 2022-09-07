//
// Created by noya-fangzhou on 11/13/21.
//

#include "AccessGraphAnalysisUtils.h"

#define DEBUG_TYPE	"accgraph-util"


AccGraph::AccGraph(SPSTNode *Root)
{
  this->Root = Root;
  QueryEngine = new TreeStructureQueryEngine(Root);
  BuildAccGraphImpl(this->Root);
}

void AccGraph::BuildAccGraphImpl(SPSTNode *Root)
{
  vector<SPSTNode *> NodesInOrder;
  GetRefTNodeInTopologyOrder(Root, NodesInOrder);
  LLVM_DEBUG(dbgs() << "Print all nodes in topology order\n");
  for (auto node : NodesInOrder) {
    LLVM_DEBUG(
        node->dump(););
    if (GraphNodeSet.find(node) == GraphNodeSet.end()) {
      GraphNodeSet.insert(node);
    }
  }
  if (NodesInOrder.empty())
    return;
  SPSTNode *current = nullptr, *next = nullptr;
  auto node_iter = NodesInOrder.begin();
  current = *node_iter;
  while (true) {
    node_iter++;
    if (node_iter != NodesInOrder.end())
      next = *(node_iter);
    else
      next = nullptr;

    LLVM_DEBUG(
        dbgs() << "Current: ";
        current->dump();
        dbgs() << "Next: ";
        if (next) {
          next->dump();
        } else {
            dbgs() << "NULL";
        }
        dbgs() << "\n";
        );
    // 1) both current and next are RefTNode type
    // 2) current is a RefTNode, next is a BranchTNode
    // 3) current is a BranchTNode, next is a RefTNode
    // 4) both current and next are BranchTNode

    // current is an access node
    if (dynamic_cast<RefTNode *>(current)) {
      RefTNode *CurrRef = dynamic_cast<RefTNode *>(current);
      if (QueryEngine->isLastAccessInLoop(current)) {
        // iterate all previous loop nodes, from inner to outer
        // for each loop, we found its first access node.
        // Then there exists a backedge that connect current with this first access
        // node.
        //
        // Note:
        // 1. if next && path to current only has 1 loop (outermost)
        // we do not add any backedges
        // 2. if next && path to curent contains multiple loops, we build
        // backedge for those inner loops (except the last one)
        // 3. if !next, we build backedge with the outermost loop
        AccPath Path;
        QueryEngine->GetPath(current, this->Root, Path, false);
        LLVM_DEBUG(
            dbgs() << Path.size() << " pathes to Root\n");
        if (!next || Path.size() > 1) { // case 2, 3
          auto path_iter = Path.begin();
          SmallVector<SPSTNode*, 8> firstAAs, lastAAs;
          while (path_iter != Path.end()) {
            LoopTNode *L = dynamic_cast<LoopTNode *>(*path_iter);
            LLVM_DEBUG(dbgs() << L->getLoopStringExpr() << " -- ");
            if (next && path_iter == Path.end() - 1) { // case 2
              break;
            }
            lastAAs = QueryEngine->GetLastAccessInLoop(L);
            firstAAs = QueryEngine->GetFirstAccessInLoop(L);
            if (find(lastAAs.begin(), lastAAs.end(), current) != lastAAs.end()) {
              // build all backedges
              for (auto firstAA : firstAAs) {
                assert(firstAA && "first access cannot be null");
                AccGraphEdge *backEdge = new AccGraphEdge(CurrRef, firstAA, L);
                this->GraphEdgeSet.emplace_back(backEdge);
                LLVM_DEBUG(
                    dbgs() << "Build a backedge between " << CurrRef->getName()
                           << " and " << firstAA->getName() << " with carry loop " << L->getLoopStringExpr() << "\n");
              }
            }
            path_iter++;
          }
        }
      }
      if (next) {
        AccGraphEdge *edge = new AccGraphEdge(current, next, NULL);
        this->GraphEdgeSet.emplace_back(edge);
        LLVM_DEBUG(
            if (edge->getCarryLoop()) {
              dbgs() << "Build an edge between " << current->getName() << " and " << next->getName() << "with carry loop " << edge->getCarryLoop()->getLoopStringExpr() << "\n";
            } else {
              dbgs() << "Build an edge between " << current->getName() << " and " << next->getName() << "\n";
            }
            );
        current = next;
        continue;
      }
    } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(current)) {
      // this is a branch node, currently, the GraphNodeSet does not include
      // nodes inside each branch.
      // for a branch node, we need to take care two things
      // 1. we need to build an edge between the last access of each branch
      // with the next
      // 2. we need to build all edges inside each branch

      // we need to connect the branch with the first access node in each
      // branch
      SmallVector<SPSTNode*, 8> firstAccInBranch;
      QueryEngine->FindFirstAccessNodesOfParent(Branch->neighbors[0], firstAccInBranch);
      for (auto firstAcc : firstAccInBranch) {
        AccGraphEdge *branchEdge = new AccGraphEdge(Branch, firstAcc, NULL);
        this->GraphEdgeSet.emplace_back(branchEdge);
        LLVM_DEBUG(
            dbgs() << "Build an edge between true branch " << Branch->getName()
                   << " and " << firstAcc->getName() << "\n");
      }
      firstAccInBranch.clear();
      // we add if condition to avoid empty false branch
      if (!Branch->neighbors[1]->neighbors.empty()) {
        QueryEngine->FindFirstAccessNodesOfParent(Branch->neighbors[1],
                                                  firstAccInBranch);
        for (auto firstAcc : firstAccInBranch) {
          AccGraphEdge *branchEdge = new AccGraphEdge(Branch, firstAcc, NULL);
          this->GraphEdgeSet.emplace_back(branchEdge);
          LLVM_DEBUG(dbgs() << "Build a branch between false branch "
                            << Branch->getName() << " and "
                            << firstAcc->getName() << "\n");
        }
      }

      // calling the BuildAccGraphImpl, we build all edges inside each branch
      // we add if condition to avoid empty false branch
      BuildAccGraphImpl(Branch->neighbors[0]);
      if (!Branch->neighbors[1]->neighbors.empty())
        BuildAccGraphImpl(Branch->neighbors[1]);
      // we need to find the last access node of each branch and build an edge
      // with the next
      SmallVector<SPSTNode*, 8> lastAccInBranch;
      QueryEngine->FindLastAccessNodesOfParent(Branch, lastAccInBranch);
      AccPath  Path;
      QueryEngine->GetPath(Branch, this->Root, Path, false);
      LLVM_DEBUG(
          dbgs() << Path.size() << " pathes to Root\n");
      auto path_iter = Path.begin();
      if (!next || Path.size() > 1) {
        SmallVector<SPSTNode*, 8> firstAAs, lastAAs;
        while (path_iter != Path.end()) {
          LoopTNode *L = dynamic_cast<LoopTNode *>(*path_iter);
          LLVM_DEBUG(dbgs() << L->getLoopStringExpr() << " -- ");
          if (next && path_iter == Path.end() - 1) { // case 2
            break;
          }
          for (auto lastAcc : lastAccInBranch) {
            firstAAs = QueryEngine->GetFirstAccessInLoop(L);
            lastAAs = QueryEngine->GetLastAccessInLoop(L);
            if (find(lastAAs.begin(), lastAAs.end(), current) != lastAAs.end()) {
              // build all backedges
              for (auto firstAA : firstAAs) {
                assert(firstAA && "first access cannot be null");
                AccGraphEdge *backEdge = new AccGraphEdge(lastAcc, firstAA, L);
                this->GraphEdgeSet.emplace_back(backEdge);
                LLVM_DEBUG(dbgs() << "Build a backedge between "
                                  << lastAcc->getName() << " and "
                                  << firstAA->getName() << " with carry loop "
                                  << L->getLoopStringExpr() << "\n");
              }
            }
          }
          path_iter++;
        }
      }
      if (next) {
        // we build a edge between the last access in each branch with the next
        // access
        for (auto lastAcc : lastAccInBranch) {
          AccGraphEdge *edge = new AccGraphEdge(lastAcc, next, NULL);
          this->GraphEdgeSet.emplace_back(edge);
          LLVM_DEBUG(
              dbgs() << "Build an edge between " << lastAcc->getName()
                     << " and " << next->getName() << "\n");
        }
        current = next;
        continue;
      }
    } // end of checking branch node
    // only if current is the last access node will reach here
    break;
  } // end of while (true)
}

void AccGraph::GetRefTNodeInTopologyOrder(SPSTNode *Root, vector<SPSTNode *> &Nodes)
{
  // iterate the Abstraction Tree using the DFS
  if (dynamic_cast<RefTNode *>(Root)) {
    Nodes.push_back(Root);
  } else if (dynamic_cast<LoopTNode *>(Root)) {
    auto neighbor_iter = Root->neighbors.begin();
    while (neighbor_iter != Root->neighbors.end()) {
      SPSTNode *Next = *neighbor_iter;
      GetRefTNodeInTopologyOrder(Next, Nodes);
      neighbor_iter++;
    }
  } else if (dynamic_cast<BranchTNode *>(Root)) {
    Nodes.push_back(Root);
    /*
    BranchTNode *Br = dynamic_cast<BranchTNode *>(Root);
    SPSTNode *TrueBranch = Br->getNextNeighbor();
    SPSTNode *FalseBranch = Br->getNextNeighbor();
    // traverse the if-branch first, then else branch
    while (TrueBranch->hasNextNeighbor()) {
      SPSTNode *Next = TrueBranch->getNextNeighbor();
      GetRefTNodeInTopologyOrder(Next);
    }

    while (FalseBranch->hasNextNeighbor()) {
      SPSTNode *Next = TrueBranch->getNextNeighbor();
      GetRefTNodeInTopologyOrder(Next);
    }
     */
  } else if (dynamic_cast<DummyTNode *>(Root)) {
    auto neighbor_iter = Root->neighbors.begin();
    for(; neighbor_iter != Root->neighbors.end(); neighbor_iter++) {
      GetRefTNodeInTopologyOrder(*neighbor_iter, Nodes);
    }
  }
}

void AccGraph::GetEdgeWithSink(SPSTNode *Sink,
                               SmallVectorImpl<AccGraphEdge *> &Edges)
{
  for (auto edge : GraphEdgeSet) {
    if (edge->hasSink(Sink))
      Edges.push_back(edge);
  }
}

void AccGraph::GetEdgesWithSource(SPSTNode *Src,
                                  SmallVectorImpl<AccGraphEdge *> &Edges)
{
  for (auto edge : GraphEdgeSet) {
    if (edge->hasSource(Src))
      Edges.push_back(edge);
  }
}

void AccGraph::GetPathToNode(SPSTNode *From, SPSTNode *To,
                             AccPath &Path)
{
  assert(From && To && "The query src and sink node must exist");
  QueryEngine->GetPath(From, To, Path, false);
}

void AccGraph::dump()
{
  vector<SPSTNode *> NodesInGraph;
  GetRefTNodeInTopologyOrder(Root, NodesInGraph);
  auto node_iter = NodesInGraph.begin();
  while (node_iter != NodesInGraph.end()) {
    if (RefTNode *Ref = dynamic_cast<RefTNode *>(*node_iter)) {
      dbgs() << Ref->getRefExprString();
    } else if (BranchTNode *Branch = dynamic_cast<BranchTNode *>(*node_iter)) {
      dbgs() << Branch->getConditionExpr();
    } else {
      llvm_unreachable("Nodes inside the AccGraph should be RefTNode or "
                       "BranchTNode type");
    }
    node_iter++;
  }
}
