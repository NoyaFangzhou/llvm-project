//
// Created by noya-fangzhou on 12/1/21.
//

#include "llvm/ADT/Statistic.h"
#include "SamplerCodeGenUtils.h"
#define DEBUG_TYPE  "sampler-codegen-utils"

cl::opt<int> SamplingTechnique(
    "sample-type", cl::init(0), cl::NotHidden,
    cl::desc("The sampling approach:\n"
             "\t0: No sampling (will ignore the sampling ratio parameter); \n"
             "\t1: Serial Start sampling; \n"
             "\t2: Random Start sampling; (only static scheduling);\n"
             "\t3: Bursty sampling; \t"));

cl::opt<double> SamplingRatio(
    "sample-rate", cl::init(100), cl::NotHidden,
    cl::desc("The sampling ratio in percentage unit, i.e. 10 means 10%"));

cl::opt<int> InterleavingTechnique(
    "interleaving-type", cl::init(0), cl::NotHidden,
    cl::desc("The thread interleaving to simulate:\n"
             "\t0: Uniform Interleaving; \n"
             "\t1: Random Interleaving; \n"));

cl::opt<bool> EnableParallelOpt(
    "parallel-opt-enable", cl::init(false), cl::NotHidden,
    cl::desc("Enable the parallelization optimization"));

cl::opt<bool> EnableModelOpt(
    "model-opt-enable", cl::init(false), cl::NotHidden,
    cl::desc("Enable the model optimization"));

cl::opt<bool> EnablePerReference(
    "per-reference-enable", cl::init(false), cl::NotHidden,
    cl::desc("Generate sampler for each references"));

STATISTIC(NumStaticComputableLoop, "Number of loops whose sample count can be "
                                   "statically derived");
STATISTIC(NumSampleCntNeedRuntime, "Number of loops whose sample count can be "
                                   "derived with programmer hint");

string SamplerCodeGenerator::tabGen()
{
  ostringstream repeated;
  fill_n(ostream_iterator<string>(repeated), tab_count, string(TAB));
//  LLVM_DEBUG(dbgs() << "repeated: " << repeated.str() << "\n");
  return repeated.str();
}


void SamplerCodeGenerator::EmitChunkDispatcher(SchedulingType type)
{
  errs() << "class ChunkEngine {\n";
  errs() << "    int lb = 0;\n";
  errs() << "    int ub = 0;\n";
  errs() << "    int chunk_size = 0;\n";
  errs() << "    int trip = 0;\n";
  errs() << "    int avail_chunk = 0;\n";
  errs() << "public:\n";
  errs() << "    ChunkEngine() {} \n";
  errs() << "    ChunkEngine(int chunk_size, int trip) {\n";
  errs() << "        assert(chunk_size <= trip);\n";
  errs() << "        this->chunk_size = chunk_size;\n";
  errs() << "        this->trip = trip;\n";
  errs() << "        this->avail_chunk = (trip / chunk_size + (trip % chunk_size));\n";
  errs() << "        this->lb = 0;\n";
  errs() << "        this->ub = chunk_size - 1;\n";
  errs() << "    }\n";
  errs() << "    string getCurrentChunkRange() {\n";
  errs() << "        return \"[\" + to_string(this->lb) + \", \" + to_string(this->ub) + \"]\";\n";
  errs() << "    }\n";
  errs() << "    bool hasNextChunk() {\n";
  errs() << "        return this->avail_chunk > 0;\n";
  errs() << "    }\n";
  errs() << "    Chunk getNextChunk(int tid) {\n";
  errs() << "        // assign the current lb, ub to thread tid and update the next chunk\n";
  errs() << "        Chunk curr = make_pair(this->lb, this->ub);\n";
  errs() << "        this->lb = this->ub + 1;\n";
  errs() << "        this->ub = (this->lb + chunk_size - 1) <= this->trip ? (this->lb + chunk_size - 1) : this->trip;\n";
  errs() << "        this->avail_chunk -= 1;\n";
  errs() << "        return curr;\n";
  errs() << "    }\n";
  errs() << "};\n";
}

void SamplerCodeGenerator::EmitProgressClassDef()
{
  errs() << "typedef pair<int, int> Chunk;\n";
  errs() << "class Progress {\n";
  errs() << "public:\n";
  errs() << "    string ref;\n";
  errs() << "    Chunk chunk;\n";
  errs() << "    vector<int> iteration;\n";
  errs() << "    Progress() { }\n";
  errs() << "    Progress(string ref, vector<int> iteration, Chunk c) {\n";
  errs() << "        this->ref = ref;\n";
  errs() << "        this->iteration = iteration;\n";
  errs() << "        this->chunk = c;\n";
  errs() << "    }\n";
  errs() << "    string getIteration() {\n";
  errs() << "        string ret = \"(\";\n";
  errs() << "        for (int i = 0; i < this->iteration.size(); i++) {\n";
  errs() << "            ret += to_string(this->iteration[i]);\n";
  errs() << "            if (i != this->iteration.size() - 1)\n";
  errs() << "                ret += \",\";\n";
  errs() << "            }\n";
  errs() << "        ret += \")\";\n";
  errs() << "        return ret;\n";
  errs() << "    }\n";
  errs() << "    string getReference() {\n";
  errs() << "        return this->ref;\n";
  errs() << "    }\n";
  errs() << "    void increment(string ref, vector<int> iteration) {\n";
  errs() << "        this->ref = ref;\n";
  errs() << "        this->iteration = iteration;\n";
  errs() << "    }\n";
  errs() << "    void increment(string ref) {\n";
  errs() << "        this->ref = ref;\n";
  errs() << "    }\n";
  errs() << "    bool isInBound() {\n";
  errs() << "        assert(this->iteration[0] >= chunk.first);\n";
  errs() << "        return this->iteration[0] <= chunk.second;\n";
  errs() << "    }\n";
  errs() << "};\n";
}

void SamplerCodeGenerator::EmitFunctionCall(string name, vector<string> params)
{
  string call = name + "(";
  for (auto param : params) {
    call += param;
    if (param != params.back()) {
      call += ",";
    }
  }
  call += ");";
  EmitCode(call);
}

void SampleNumberAnalyzer::CalculateSampleNumberForLoop(LoopTNode *LoopNode,
                                                        vector<LoopTNode *> Loops) {
  vector<LoopTNode *> path = Loops;
  path.push_back(LoopNode);
  bool AllLoopHasConstantBound = true;

  vector<bool> isLoopLBConstant(path.size()), isLoopUBConstant(path.size());
  unsigned level = 0;
  for (; level < path.size(); level++) {
    // check constant bounds for each loop inside the loop nests
    LoopBound *LB = path[level]->getLoopBound();
    if (LB && LB->FinalValue) {
      bool isInitValueConstant = isa<ConstantInt>(LB->InitValue);
      bool isFinalValueConstant = isa<ConstantInt>(LB->FinalValue);
      isLoopLBConstant[level] = isInitValueConstant;
      isLoopUBConstant[level] = isFinalValueConstant;
      AllLoopHasConstantBound &= (isInitValueConstant && isFinalValueConstant);
    } else {
      isLoopLBConstant[level] = false;
      isLoopUBConstant[level] = false;
      AllLoopHasConstantBound = false;
    }
  }
  if (AllLoopHasConstantBound) {
    unsigned long trip_count = 1;
    // all loops in this loop nest has constant loop bound
    // we go over these loop from outer to inner and the total iteration of this
    // loop nest is the multiple of their loop tirp
    for (auto loop : path) {
      LoopBound *LB = loop->getLoopBound();
      assert(LB && "LoopBound object should not be nullptr");
      int init_value = dyn_cast<ConstantInt>(LB->InitValue)->getSExtValue();
      int final_value = dyn_cast<ConstantInt>(LB->FinalValue)->getSExtValue();
      int step = dyn_cast<ConstantInt>(LB->StepValue)->getSExtValue();
      int loop_trip_count = abs(final_value - init_value);
      switch (LB->Predicate) {
      case llvm::CmpInst::ICMP_SLE:
      case llvm::CmpInst::ICMP_SGE:
      case llvm::CmpInst::ICMP_ULE:
      case llvm::CmpInst::ICMP_UGE:
        loop_trip_count += 1;
        break;
      default:
        break;
      }
      trip_count *= (loop_trip_count / abs(step));
    }
    int sample_num = ceil(trip_count * SamplingRate / 100.);
    for (unsigned i = 0; i < Loops.size(); i++) {
      sample_num = ceil(sample_num * SamplingRate / 100.);
    }
    PerLoopSampleNumTable[LoopNode] = sample_num;
    NumStaticComputableLoop++;
  } else if (!LoopNode->getLoopBound()) {
    LLVM_DEBUG(dbgs() << "The trip of " << LoopNode->getLoopStringExpr()
                      << " cannot be statically computed");
    NumSampleCntNeedRuntime++;
  } else if (isa<Argument>(LoopNode->getLoopBound()->InitValue)
             || isa<Argument>(LoopNode->getLoopBound()->FinalValue)) {
    LLVM_DEBUG(dbgs() << "The trip of " << LoopNode->getLoopStringExpr()
                      << " cannot be statically computed, need hint from programmer");
    NumSampleCntNeedRuntime++;
  } else {
    unsigned long trip_count = 0;
    TranslateStatus status = SUCCESS;
    // some loop init/final value is depends on the induction variable of one
    // of its parent loop


    // this is an optimization
    // if the current innermost loop is constant bound, and its immediate loop
    // dominator's trip has already been derived
    // the trip cound of the loop nest in path would be
    // trip_count = innermost_loop_trip_count * PerLoopSampleNumTable[path[level-1]]
    level = path.size()-1;
    if (level-1 < path.size() && isLoopLBConstant[level] &&
        isLoopUBConstant[level] &&
        PerLoopSampleNumTable.find(path[level-1]) != PerLoopSampleNumTable.end()) {
      LoopBound *InnermostLoopLB = path[level]->getLoopBound();
      int last_level_ub = dyn_cast<ConstantInt>(InnermostLoopLB->FinalValue)->getSExtValue();
      int last_level_lb = dyn_cast<ConstantInt>(InnermostLoopLB->InitValue)->getSExtValue();
      int step = dyn_cast<ConstantInt>(InnermostLoopLB->StepValue)->getSExtValue();
      int innermost_trip_count = abs(last_level_ub - last_level_lb) / abs(step);
      trip_count = PerLoopSampleNumTable[path[level-1]] * innermost_trip_count;

      // we do not have to multiply the Sampling ratio again because we use
      // the sample number from upper loop level
      // PerLoopSampleNumTable[path[level-1]] = parent_loop_trip * SampleRate
      // PerLoopSampleNumTable[path[level]] = parent_loop_trip * innermostloop_trip * SampleRate
      PerLoopSampleNumTable[LoopNode] = ceil(trip_count * SamplingRate / 100.);;
      NumStaticComputableLoop++;
    } else {

      // counter : (Value * -> int)     the constant value of each induction variable
      unordered_map<Value *, int> counter;

      // init the counter
      for (level = 0; level < path.size(); level++) {
        if (level ==
            0) { // the outermost loop nest should have a constant loop bound
          int loop_init_val =
              dyn_cast<ConstantInt>(path[level]->getLoopBound()->InitValue)
                  ->getSExtValue();
          counter[path[level]->getInductionPhi()] = loop_init_val;
        } else if (isLoopLBConstant[level]) {
          int loop_init_val =
              dyn_cast<ConstantInt>(path[level]->getLoopBound()->InitValue)
                  ->getSExtValue();
          counter[path[level]->getInductionPhi()] = loop_init_val;
        } else {
          counter[path[level]->getInductionPhi()] = ReplaceValueWithConstant(
              path[level]->getLoopBound()->InitValue, counter, status);
        }
      }

      vector<int> strides(path.size());
      // init the step of each loop
      for (level = 0; level < path.size(); level++) {
        int step = dyn_cast<ConstantInt>(path[level]->getLoopBound()->StepValue)
                       ->getSExtValue();
        // strides can be positive/negative, need to handle them well in later
        // sample computation part
        strides[level] = step;
      }

      level = path.size() - 1; // starts from the innermost loop

      while (true) {
        int last_level_ub = ReplaceValueWithConstant(
            path[level]->getLoopBound()->FinalValue, counter, status);
        int last_level_lb = ReplaceValueWithConstant(
            path[level]->getLoopBound()->InitValue, counter, status);
        //      LLVM_DEBUG(dbgs() << abs(last_level_lb - last_level_ub) << " + " );
        int last_level_trip_count = abs(last_level_lb - last_level_ub);
        if ((path[level]->getLoopBound()->Predicate ==
             llvm::ICmpInst::ICMP_UGE) ||
            (path[level]->getLoopBound()->Predicate ==
             llvm::ICmpInst::ICMP_SGE)) {
          last_level_trip_count += 1;
        }
        trip_count += (last_level_trip_count / abs(strides[level]));

        LLVM_DEBUG(dbgs() << "init counter \n";
                   for (int i = path.size() - 1; i >= 0; i--) {
                     dbgs() << "counter[" << i << "] "
                            << counter[path[i]->getInductionPhi()] << "\n";
                   });

        while (level - 1 < path.size()) {
          level -= 1;
          bool inside_loop_bound =
              true; // jump out of the while (level - 1 >= 0) loop
          // increment the expression at level 'level' with strides[level]
          int increment = ReplaceValueWithConstant(
              path[level]->getLoopBound()->StepInst, counter, status);
          counter[path[level]->getInductionPhi()] = increment;
          int loop_final_val = ReplaceValueWithConstant(
              path[level]->getLoopBound()->FinalValue, counter, status);
          LLVM_DEBUG(for (int i = path.size() - 1; i >= 0; i--) {
            dbgs() << "counter[" << i << "] "
                   << counter[path[i]->getInductionPhi()] << "\n";
          });

          switch (path[level]->getLoopBound()->Predicate) {
          case llvm::CmpInst::ICMP_UGE:
          case llvm::CmpInst::ICMP_SGE: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] >= loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val >= counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_ULE:
          case llvm::CmpInst::ICMP_SLE: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] <= loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val <= counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_UGT:
          case llvm::CmpInst::ICMP_SGT: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] > loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val > counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_ULT:
          case llvm::CmpInst::ICMP_SLT: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] < loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val < counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_NE: {
            inside_loop_bound =
                counter[path[level]->getInductionPhi()] != loop_final_val;
            break;
          }
          default:
            break;
          }
          if (inside_loop_bound)
            break;
        }

        if (level == 0) {
          bool inside_loop_bound = true; // jump out of the while(true) loop
          int loop_final_val = ReplaceValueWithConstant(
              path[level]->getLoopBound()->FinalValue, counter, status);
          switch (path[level]->getLoopBound()->Predicate) {
          case llvm::CmpInst::ICMP_UGE:
          case llvm::CmpInst::ICMP_SGE: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] >= loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val >= counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_ULE:
          case llvm::CmpInst::ICMP_SLE: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] <= loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val <= counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_UGT:
          case llvm::CmpInst::ICMP_SGT: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] > loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val > counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_ULT:
          case llvm::CmpInst::ICMP_SLT: {
            if (path[level]->getLoopBound()->isLHS())
              inside_loop_bound =
                  counter[path[level]->getInductionPhi()] < loop_final_val;
            else
              inside_loop_bound =
                  loop_final_val < counter[path[level]->getInductionPhi()];
            break;
          }
          case llvm::CmpInst::ICMP_NE: {
            inside_loop_bound =
                counter[path[level]->getInductionPhi()] != loop_final_val;
            break;
          }
          default:
            break;
          }
          if (!inside_loop_bound)
            break;
        }

        // update the value of the induction variable of all inner loops
        level++;
        while (level < path.size()) {
          int loop_init_val = ReplaceValueWithConstant(
              path[level]->getLoopBound()->InitValue, counter, status);
          counter[path[level]->getInductionPhi()] = loop_init_val;
          level++;
        }
        LLVM_DEBUG(dbgs() << "reset counter \n";
                   for (int i = path.size() - 1; i >= 0; i--) {
                     dbgs() << "counter[" << i << "] "
                            << counter[path[i]->getInductionPhi()] << "\n";
                   });
        level = path.size() - 1; // go back to the innermost loop nests
      } // end of while(true)
      int sample_num = ceil(trip_count * SamplingRate / 100.);
      for (unsigned i = 0; i < Loops.size(); i++) {
        sample_num = ceil(sample_num * SamplingRate / 100.);
      }
      PerLoopSampleNumTable[LoopNode] = sample_num;
      NumStaticComputableLoop++;
    }

    LLVM_DEBUG(dbgs() << "Loop " << LoopNode->getLoopStringExpr()
                      << " has " << PerLoopSampleNumTable[LoopNode]
                      << " samples\n");
  }

  if (!LoopNode->neighbors.empty()) {
    Loops.push_back(LoopNode);
    for (auto child : LoopNode->neighbors) {
      if (LoopTNode *ChildLoop = dynamic_cast<LoopTNode *>(child)) {
        CalculateSampleNumberForLoop(ChildLoop, Loops);
      }
    }
  }
  LLVM_DEBUG(dbgs() << "\n");

#if 0
  LoopBound *LB = LoopNode->getLoopBound();
  bool isInitValueConstant = isa<ConstantInt>(LB->InitValue);
  bool isFinalValueConstant =
      ((LB->FinalValue != nullptr) && isa<ConstantInt>(LB->FinalValue));
  unsigned sample_count = 0;
  if (isInitValueConstant && isFinalValueConstant) {
    int init_value = dyn_cast<ConstantInt>(LB->InitValue)->getSExtValue();
    int final_value = dyn_cast<ConstantInt>(LB->FinalValue)->getSExtValue();
    int step = dyn_cast<ConstantInt>(LB->StepValue)->getSExtValue();
    unsigned trip_count = 0;

    switch(LB->Predicate) {
    case llvm::CmpInst::ICMP_SLE:
    case llvm::CmpInst::ICMP_SGE:
    case llvm::CmpInst::ICMP_ULE:
    case llvm::CmpInst::ICMP_UGE:
      trip_count = (final_value - init_value + 2) / abs(step);
      break;
    case llvm::CmpInst::ICMP_SLT:
    case llvm::CmpInst::ICMP_SGT:
    case llvm::CmpInst::ICMP_ULT:
    case llvm::CmpInst::ICMP_UGT:
      trip_count = (final_value - init_value + 1) / abs(step);
      break;
    }
    LLVM_DEBUG(
        dbgs() << "Loop " << LoopNode->getLoopStringExpr()
               << " has " << trip_count << " iterations\n";
        );
    PerLoopSampleCount[LoopNode] = trip_count;
  } else {
    // we will recover the loop that its induction depends on
    // then its trip will be computed
    // i.e. for (i = 0; i < j; i++)
    // we will recover the j loop first and its trip count will be
    // for (j = 0; j < 0; j++)
    //  trip_count += (j - 0 + 1) / 1;
    // Note that the for-loop will be recovered until we met a loop that has
    // a constant bound

    unsigned level = 0; // i indicate the level of the loop nest to the target loop



  }

  for (auto neighbor: LoopNode->neighbors) {
    if (LoopTNode *ChildLoopNode = dynamic_cast<LoopTNode *>(neighbor)) {
      CalculateSampleNumberForLoop(ChildLoopNode, PerLoopSampleCount);
    }
  }
#endif
}

unsigned SampleNumberAnalyzer::getSampleNumberForRef(RefTNode *RefNode)
{
  AccPath path;
  this->Engine->GetPath(RefNode, nullptr, path);
  LLVM_DEBUG(dbgs() << "Compute SampleNumber for Ref " << RefNode->getRefExprString() << "\n");
  assert(!path.empty() && "The given RefNode should inside a loop nest");
  // backtrack all LoopNodes from the RefNode
  //
  // for sequential, its sample number would the number of samples found by
  // its outermost loop (level == 0)
  //
  // for parallel, its sample number would be the number of samples found by
  // the parallel loop (level where path[level] is a parallel loop
#if 0
  while (pathIter != path.rend()) {
    LoopTNode *LoopNode = dynamic_cast<LoopTNode *>(*pathIter);
    assert(LoopNode && "Element inside a path should be an instance of LoopTNode "
                       "class");
    if (LoopNode->isParallelLoop()) {
      TargetLoop = LoopNode;
      break;
    }
    pathIter++;
  }
#endif
//  CalculateSampleNumberForLoop(dynamic_cast<LoopTNode *>(*(path.rbegin())), {});
  LoopTNode *IDom = Engine->GetImmdiateLoopDominator(RefNode);
  if (IDom && PerLoopSampleNumTable.find(IDom) != PerLoopSampleNumTable.end()) {
    return PerLoopSampleNumTable[IDom];
  } else {
    LLVM_DEBUG(
        dbgs() << "Reference " << RefNode->getRefExprString()
               << " is inside a loop nest whose sampling number cannot be "
                  "computed during compile-time");
  }
  return 0;
}

void SampleNumberAnalyzer::analyze() {
  for (auto node : Root->neighbors) {
    if (LoopTNode *Loop = dynamic_cast<LoopTNode *>(node)) {
      CalculateSampleNumberForLoop(Loop, {});
    }
  }
  LLVM_DEBUG(
  for (auto pair : PerLoopSampleNumTable) {
    dbgs() << pair.first->getLoopStringExpr() << " \t" << pair.second << "\n";
  }
  );
}



