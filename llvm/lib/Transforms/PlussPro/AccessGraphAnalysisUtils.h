//
// Created by noya-fangzhou on 11/13/21.
//

#ifndef LLVM_ACCESSGRAPHANALYSISUTILS_H
#define LLVM_ACCESSGRAPHANALYSISUTILS_H

#include "LoopAnalysisUtils.h"
#include "PlussUtils.h"

class AccGraphEdge {
  SPSTNode *Src;
  SPSTNode *Sink;
  LoopTNode *CarryLoop; // NULL means forward edge, else backward edge
public:
  AccGraphEdge();
  AccGraphEdge(SPSTNode *Src, SPSTNode *Sink, LoopTNode *L) : Src(Src),
                                                              Sink(Sink),
                                                              CarryLoop(L) {};
  LoopTNode *getCarryLoop() { return this->CarryLoop; }
  SPSTNode *getSrc() { return this->Src; }
  SPSTNode *getSink() { return this->Sink; }
  bool hasSource(SPSTNode *node) { return this->Src == node; }
  bool hasSink(SPSTNode *node) { return this->Sink == node; }
  bool containsNode(SPSTNode * node) { return hasSource(node) || hasSink(node); }
  bool operator==(const AccGraphEdge &rhs) {
    return (this->Src == rhs.Src && this->Sink == rhs.Sink &&
           this->CarryLoop == rhs.CarryLoop);
  }
  bool operator<(const AccGraphEdge &rhs) const {
    if (this->CarryLoop && !rhs.CarryLoop)
      return true;
    else if (this->CarryLoop && rhs.CarryLoop) {
      return this->CarryLoop->getLoopLevel() < rhs.CarryLoop->getLoopLevel();
    }
    return true;
  }
};

class AccGraph {
  SPSTNode *Root;
  set<SPSTNode *> GraphNodeSet;
  vector<AccGraphEdge *> GraphEdgeSet;
public:
  TreeStructureQueryEngine *QueryEngine;
  AccGraph();
  AccGraph(SPSTNode *);
  SPSTNode *getRoot() { return this->Root; }
  set<SPSTNode *> GetAllNodesInGraph() { return this->GraphNodeSet; }
  vector<AccGraphEdge *> GetAllGraphEdges() { return this->GraphEdgeSet; }
  void GetPathToNode(SPSTNode *, SPSTNode *, AccPath &);
  void GetEdgesWithSource(SPSTNode *, SmallVectorImpl<AccGraphEdge *>&);
  void GetEdgeWithSink(SPSTNode *, SmallVectorImpl<AccGraphEdge *>&);
  void GetRefTNodeInTopologyOrder(SPSTNode *, vector<SPSTNode *> &);
  unsigned GetNumEdges() { return this->GraphEdgeSet.size(); }
  unsigned GetNumNodes() { return this->GraphNodeSet.size(); }
  void dump();
private:
  void BuildAccGraphImpl(SPSTNode *);
  unordered_map<SPSTNode *, SPSTNode *> ParentNodeMapping;

};
#endif // LLVM_ACCESSGRAPHANALYSISUTILS_H
