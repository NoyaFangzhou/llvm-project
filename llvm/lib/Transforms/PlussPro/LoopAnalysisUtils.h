//
//  LoopAnalysisUtils.hpp
//  LLVMSPSAnalysis
//
//  Created by noya-fangzhou on 10/14/21.
//

#ifndef LoopAnalysisUtils_H
#define LoopAnalysisUtils_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "PlussUtils.h"

using namespace llvm;
using namespace std;

enum IVPos {
  LHS,
  RHS,
  UNKNOWN
};

class LoopBound {
  IVPos InductionVariablePos;
public:
  Value *InitValue;
  Value *FinalValue;
  const SCEV *FinalValueSCEV;
  Value *StepValue;
  Instruction *StepInst;
  CmpInst::Predicate Predicate;
  LoopBound(Value *InitIVValue, const SCEV *FinalValueSCEV, Value *StepValue,
            Instruction *StepInst, CmpInst::Predicate P, Value *FinalIVValue=nullptr, IVPos pos=LHS) {
    this->InitValue = InitIVValue;
    if (FinalIVValue)
      this->FinalValue = FinalIVValue;
    this->FinalValueSCEV = FinalValueSCEV;
    this->StepValue = StepValue;
    this->StepInst = StepInst;
    this->Predicate = P;
    this->InductionVariablePos = pos;
  }
  bool isLHS() { return InductionVariablePos == LHS; }
};

class SPSTNode {
protected:
  string name;
public:
  vector<SPSTNode *> neighbors;
  SPSTNode() {
    name = "SPSTNode";
  }
  virtual void dump() = 0;
  virtual ~SPSTNode() {}
  string getName() const { return name; }
  void addNeighbors(SPSTNode *Node) {
    this->neighbors.push_back(move(Node));
  }
  bool hasNeighbor() { return !this->neighbors.empty(); }
  void setNeighbors(vector<SPSTNode *> newNeighbors) {
    this->neighbors.clear();
    vector<SPSTNode *>::iterator it = newNeighbors.begin();
    for (; it != newNeighbors.end(); ++it) {
      this->neighbors.push_back(*it);
    }
  }
  unsigned getNumNeighbors() { return this->neighbors.size(); }
  bool eraseFromNeighbor(SPSTNode *Node) {
    if (find(neighbors.begin(), neighbors.end(), Node) == neighbors.end())
      return false;
    neighbors.erase(
        remove(neighbors.begin(), neighbors.end(), Node),
        neighbors.end());
    return true;
  }
};

class DummyTNode : public SPSTNode {
public:
  DummyTNode() : SPSTNode() {}
  DummyTNode(string name) : SPSTNode() {
    this->name = name;
  }
  void dump() override {
    dbgs() << " DUMMY NODE";
  }
  ~DummyTNode() {}
};

class LoopTNode : public SPSTNode {
  unsigned LoopLevel;
  Loop *L;
  unsigned ID;
  PHINode *InductionVariable;
  bool IsParallelLoop;
  string LoopStringExpr;
  string LoopCodeExpr;
  ScalarEvolution *SE;
  LoopBound *LB;
public:
  /* getter / setter */
  unsigned getLoopID() const { return ID; }
  unsigned getLoopLevel() const { return LoopLevel; }
  void setLoopLevel(unsigned int loopLevel) { LoopLevel = loopLevel; }
  Loop *getLoop() const { return L; }
  void setLoop(Loop *l) { L = l; }
  PHINode *getInductionPhi() const { return InductionVariable; }
  bool isParallelLoop() const { return IsParallelLoop; }
  void setParallelLoop(bool isParallelLoop) {
    IsParallelLoop = isParallelLoop;
  }
  const string &getLoopStringExpr() const { return LoopStringExpr; }
  void setLoopStringExpr(const string &loopStringExpr) {
    LoopStringExpr = loopStringExpr;
  }
  void setLoopBound(LoopBound *LB) { this->LB = LB; }
  LoopBound* getLoopBound() { return this->LB; }

public:
  LoopTNode(Loop *L, ScalarEvolution &SE) : SPSTNode(), L(L) {}
  LoopTNode(Loop *L, unsigned id, bool IsParallel, ScalarEvolution &SE)
      : SPSTNode(), L(L), ID(id), IsParallelLoop(IsParallel) {}
  LoopTNode(Loop *L, unsigned LL, unsigned id, ScalarEvolution &SE);
  LoopTNode(Loop *L, unsigned LL, unsigned id, bool IsParallel,
                       ScalarEvolution &SE);
  void dump() override;
  ~LoopTNode() {}
  bool isInLoop(BasicBlock *Block) { return this->L->contains(Block); }
private:
  void ComputeLoopBound(ScalarEvolution &);
};

class ThreadTNode : public SPSTNode {
public:
  ThreadTNode() : SPSTNode() { this->name = "ThreadTNode"; }
  void dump() override {
    dbgs() << " THREAD NODE";
  }
  ~ThreadTNode() {}
};

// Its has two neighbors, TrueNode and FalseNode
// TrueNode stores the path where condition is true (taken)
// FalseNode stores the path where condition is false (not taken)
class BranchTNode : public SPSTNode {
  Value *CondInst;
  string CondExpr;
public:
  BranchTNode(Value *Cond) : SPSTNode(), CondInst(Cond) {
    this->name = "BranchTNode";
    DummyTNode *TrueNode = new DummyTNode("True");
    DummyTNode *FalseNode = new DummyTNode("False");
    this->addNeighbors(TrueNode);
    this->addNeighbors(FalseNode);
  }
  void dump() override {
    dbgs() << CondExpr << "\n";
  }
  void setCondition(Value *Cond) { CondInst = Cond; }
  Value *getCondition() { return CondInst; }
  void setConditionExpr(string expr) { CondExpr = expr; }
  string getConditionExpr() { return CondExpr; }
  void addTrueNode(SPSTNode *node) {
    this->neighbors[0]->addNeighbors(node);
  }
  void addFalseNode(SPSTNode *node) {
    this->neighbors[1]->addNeighbors(node);
  }
  ~BranchTNode() {}
};

class RefTNode : public SPSTNode {
  string arrayName;             // the array accessed
  string Expression;		// The index expression
  Value *Base;                  // The base of the array
  Instruction *MemOp;	        // The access instruction of the array
  GetElementPtrInst *ArrayAccess;     // The access instruction of the array
  vector<Value *> Subscripts;

public:
  /* getter / setter */
  void setExprString(string array, string expr) {
    this->arrayName = array;
    this->Expression = expr;
  }
  string getArrayNameString() { return this->arrayName; }
  string getRefExprString() { return (this->arrayName + this->Expression); }
  string getIndexExprString() { return this->Expression; }
  Value *getBase() const { return Base; }
  void setBase(Value *base) { Base = base; }
  Instruction *getMemOp() const { return MemOp; }
  void setMemOp(Instruction *memOp) { MemOp = memOp; }
  GetElementPtrInst *getArrayAccess() const { return ArrayAccess; }
  void setArrayAccess(GetElementPtrInst *arrayAccess) {
    ArrayAccess = arrayAccess;
  }
  const vector<Value *> &getSubscripts() const { return Subscripts; }
public:
  RefTNode(Instruction *Ref, GetElementPtrInst *GEP, ScalarEvolution &SE);

  void dump() override;
  ~RefTNode() {}
private:
  Value *FindBaseAndSubscript(Value *V, SmallVectorImpl<Value *> &Subscripts);
};

enum DIRECTION {
  LOWER_BOUND_DEP,
  UPPER_BOUND_DEP,
  DUAL_DEP,
  NO_DEP
};


class IVDepNode {
  Loop *L;
  PHINode *InductionPhi;
  unsigned Level;
public:
  SmallVector<pair<IVDepNode *, DIRECTION>, 16> dependences;
  SmallVector<pair<IVDepNode *, DIRECTION>, 16> parents;
  IVDepNode() {};
  IVDepNode(Loop *Loop, PHINode *Phi, unsigned l) : L(Loop), InductionPhi(Phi),
                                                    Level(l) {};
  void addDependency(IVDepNode *Node, DIRECTION d) {
    dependences.push_back(make_pair(Node, d));
  }
  void addParent(IVDepNode *Node, DIRECTION d) {
    parents.push_back(make_pair(Node, d));
  }
  bool isNodeOf(PHINode *Phi) { return this->InductionPhi == Phi; }
  bool isNodeOf(Loop *L) { return this->L == L; }
  DIRECTION isChildOf(Loop *L) {
    return isChildOf(L->getCanonicalInductionVariable()); }
  DIRECTION isChildOf(PHINode *Phi) {
    for (auto node : parents) {
      if (node.first->isNodeOf(Phi))
        return node.second;
    }
    return NO_DEP;
  }
  DIRECTION isParentOf(Loop *L) {
    return isParentOf(L->getCanonicalInductionVariable()); }
  DIRECTION isParentOf(PHINode *Phi) {
    for (auto node : dependences) {
      if (node.first->isNodeOf(Phi))
        return node.second;
    }
    return NO_DEP;
  }
  Loop *getLoop() { return this->L; }
  PHINode *getInduction() { return this->InductionPhi; }
  unsigned getLevel() { return Level; }
  unsigned getNumParents() { return this->parents.size(); }
  unsigned getNumDependencies() { return this->dependences.size(); }
};

Loop *FindSubloopContainsBlockInLoop(Loop *, BasicBlock *);
void FindSubLoopsContainBlock(Loop *, BasicBlock *,
                              vector<Loop *> &SubLoops);
void getBasicBlockInAllSubLoops(Loop *, unordered_set<BasicBlock *> &,
    bool ExcludeL=false);
void getBasicBlockInAllSubLoopsInOrder(Loop *L,
                                       vector<BasicBlock *> &SubLoopBasicBlocks,
                                       bool ExcludeL);
PHINode *getInductionVariable(Loop *L, ScalarEvolution *SE);
LoopBound *BuildLoopBound(Loop *L, ScalarEvolution *SE,
                          ICmpInst *LoopBranch, bool NeedInversePredicate,
                          PHINode *IndVar);
MDNode *GetMetadataWithName(MDNode *LoopMD, StringRef Name);
unsigned GetLoopMetadataConstValue(MDNode *LoopMD,
                                   StringRef Name="llvm.loop.pluss.bound");

string GetSPSTNodeName(SPSTNode *);


typedef SmallVector<SPSTNode *, 32> AccPath;

// answer some structure questions in the abstraction tree
class TreeStructureQueryEngine {
  SPSTNode *Root;
  unsigned TreeDegree;
  // key: child, value: its parent
  // child can be either a branch or a access node
  DenseMap<SPSTNode *, LoopTNode *> ParentNodeMapping;
  DenseMap<SPSTNode *, SmallVector<SPSTNode *, 8>> FirstAccesses;
  DenseMap<SPSTNode *, SmallVector<SPSTNode *, 8>> LastAccesses;
public:
  TreeStructureQueryEngine(SPSTNode *Root);
  bool isTopologicallyLargerThan(SPSTNode *A, SPSTNode *B);
  bool areAccessToSameArray(RefTNode *, RefTNode *);
  bool isReachable(SPSTNode *, RefTNode *);
  bool isFirstAccess(SPSTNode *);
  bool isLastAccessInLoop(SPSTNode *);
  bool areTwoAccessInSameLevel(SPSTNode *, SPSTNode *);
  bool areTwoAccessInSameLoop(SPSTNode *, SPSTNode *);
  bool hasBranchInside(LoopTNode *);
  bool hasConstantLoopBound(LoopTNode *);
  bool hasConstantLoopUpperBound(LoopTNode *);
  bool hasConstantLoopLowerBound(LoopTNode *);
  SmallVector<SPSTNode *, 8> GetFirstAccessInLoop(SPSTNode *);
  SmallVector<SPSTNode *, 8> GetLastAccessInLoop(SPSTNode *);

  void GetImmediateCommonLoopDominator(SPSTNode *, SPSTNode *,
                                             bool isLastAccess,
                                             SmallVectorImpl<SPSTNode *> &);
  LoopTNode *GetImmdiateLoopDominator(SPSTNode *);
  LoopTNode *GetParallelLoopDominator(SPSTNode *);
  bool isLastLevelAccess(SPSTNode *);
  void GetPath(SPSTNode *, SPSTNode *, AccPath &, bool ExcludeSink = true);
  void FindFirstAccessNodesOfParent(SPSTNode *Node,
                                    SmallVector<SPSTNode *, 8> &FirstRefs);
  void FindLastAccessNodesOfParent(SPSTNode *Node,
                                   SmallVector<SPSTNode *, 8> &LastRefs);
  void GetSubLoopTNode(SPSTNode *, SmallVectorImpl<LoopTNode *> &);
  void FindAllRefsInLoop(SPSTNode *, vector<RefTNode *> &);
private:
  void BuildParentMapping(SPSTNode *Root);

};

// some compiler optimization may change the loop condition
// to its negation form.
// i.e.
// for（i = 0; i <= j; i++) -> for (i =0; !(i > j); i++)
// this function tries to check if the loop is transformed by such optimizations
bool isLoopBranchNetativeTransformed(LoopTNode *);

#endif /* LoopAnalysisUtils_H */
