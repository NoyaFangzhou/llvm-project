//
// Created by noya-fangzhou on 12/1/21.
//

#ifndef LLVM_SAMPLERCODEGENUTILS_H
#define LLVM_SAMPLERCODEGENUTILS_H

#include "PlussUtils.h"
#include "LoopAnalysisUtils.h"

#define TAB "    "
// common types macro
#define INT_TYPE              "int"
#define UINT_TYPE             "unsigned"
#define DOUBLE_TYPE           "double"
#define FLOAT_TYPE            "float"
#define STR_TYPE              "string"
// std containers macro
#define VECTOR_TYPE(X)        "vector<" #X ">"
#define UMAP_TYPE(X,Y)        "unordered_map<" #X " ," #Y ">")
#define MAP_TYPE(X,Y)         "map<" #X " ," #Y ">")
#define SET_TYPE(X)           "set<" #X ">"
#define USET_TYPE(X)          "unordered_set<" #X ">"

enum SamplingType {
  NO_SAMPLE,
  SEQUENTIAL_START,               /* Sequential Start sampling */
  RANDOM_START,                   /* Random Start sampling */
  BURSTY                          /* Bursty sampling */
};

enum Interleaving {
  UNIFORM_INTERLEAVING,
  RANDOM_INTERLEAVING
};

enum CodeGenType {
  DEFAULT,
  OMP_PARALLEL,
  OMP_PARALLEL_INIT,
  SAMPLE,
  MODEL
};

extern cl::opt<double> SamplingRatio;

extern cl::opt<int> SamplingTechnique;

extern cl::opt<int> InterleavingTechnique;

extern cl::opt<bool> EnableParallelOpt;

extern cl::opt<bool> EnableModelOpt;

extern cl::opt<bool> EnablePerReference;

class SamplerCodeGenerator {
public:
  SamplerCodeGenerator() {};
  void EmitFunctionCall(string name, vector<string> params);
  void EmitChunkDispatcher(SchedulingType type);
  void EmitProgressClassDef();
  // automatically append the '\n' at the end of the code
  void EmitComment(string comment) {
    errs() << (tabGen() + "//" + comment) << "\n";
  }
  void EmitCode(string code) {
    errs() << (tabGen() + code) << "\n";
  }
  // automatically append the ';\n' at the end of the code
  void EmitStmt(string code)   {
    errs() << (tabGen() + code) << ";\n";
  }
  void EmitLabel(string label) { errs() << label  << ":\n"; }
  void EmitDebugInfo(string body) {
    errs() << "#if defined(DEBUG)\n";
    this->EmitStmt(body);
    errs() << "#endif\n";
  }
  void EmitHashMacro(string macro) {
    errs() << macro << "\n";
  }
#if 0
  void EmitDebugInfo(function<void(void)> body) {
    errs() << "#if defined(DEBUG)\n";
    body();
    errs() << "#endif\n";
  }
#endif
  // use to track the number of tabs we need to add
  unsigned tab_count = 0;
private:
//  stack<unsigned> tabstack;
  string tabGen();
};

class SampleNumberAnalyzer {
  double SamplingRate;
  SPSTNode *Root;
public:
  SampleNumberAnalyzer(SPSTNode *Root, double SamplingRate) {
    this->Root = Root;
    this->SamplingRate = SamplingRate;
    this->Engine = new TreeStructureQueryEngine(Root);
  }
  unordered_map<LoopTNode *, unsigned> PerLoopSampleNumTable;
  unsigned getSampleNumberForRef(RefTNode *);
  void analyze();
private:
  TreeStructureQueryEngine *Engine;
  void CalculateSampleNumberForLoop(LoopTNode *, vector<LoopTNode *>);
};

#endif // LLVM_SAMPLERCODEGENUTILS_H
