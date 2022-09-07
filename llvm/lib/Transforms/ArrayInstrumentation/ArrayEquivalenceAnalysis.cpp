//
//  ArrayEquivalenceAnalysis.cpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 4/20/21.
//
#include <stack>
#include <unordered_set>

#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Instruction.h"
#include "ArrayEquivalenceAnalysis.hpp"
#include "ArrayInstrumentationUtils.hpp"

#define DEBUG_TYPE 	"array-equal-analysis"

using namespace std;
using namespace llvm;


bool ArrayEqDetector::isProfitFunc(Function &F)
{
	// Iterate every instruction in this function
	// Profit func is the func that contains array access instructions
	for (auto &B : F) {
		for (auto &I : B) {
			if (auto *op = dyn_cast<GetElementPtrInst>(&I)) {
				return true;
			}
		}
	}
	return false;
}

namespace {
	struct ArrayEquivalenceAnalysis : public ModulePass {
		static char ID;
		ArrayEquivalenceAnalysis() : ModulePass(ID) {}
		
		static ArrayEqDetector Detector;

		/// Call for every module
		virtual bool runOnModule(Module &M) {
			
			// Iterate the module and list all functions
			stack<Function *> workingList;
			unordered_set<Function *> visited;
			for (Function &F : M) {
				// Skip indirect calls unknown to the compiler (indirect calls),
				// calls that does not contains array expression
				if (F.isDeclaration() || !Detector.isProfitFunc(F)) continue;
				workingList.push(&F);
			}
			
			CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
			Function * F = NULL;
			// Traverse the call graph to collect the function-reaching relations.
			// It is a breadth first search.
			while (!workingList.empty()) {
				do {
					F = workingList.top();
					workingList.pop();
				} while (visited.find(F) != visited.end() && !workingList.empty());
				if (visited.find(F) != visited.end()) continue;
				visited.insert(F);
				
				// Read all parameters that passed to the callee.
				for (auto &B : *F) {
					for (auto &I : B) {
						if (auto *op = dyn_cast<CallInst>(&I)) {
							for (unsigned i = 0; i < op->arg_size(); i++) {
								Value * param = getArrayNameExpr(op->User::getOperand(i));
								
							}
						}
					}
				}
				
				// stores a set of callees that F calls
				unordered_set<Function *> visitedCallee;
				unordered_set<Function *> profitCallee;
				for (CallGraphNode::iterator CGNI = CG[F]->begin(); CGNI != CG[F]->end(); CGNI++) {
					
					// The iterator gives a pair of <WeakTrackingVH, CallGraphNode *>
					Function *callee = CGNI->second->getFunction();
					LLVM_DEBUG(dbgs() << "Function " << F->getName() << " calls " << callee->getName()
							   << "\n"
							   );
					if (visitedCallee.find(callee) != visitedCallee.end()) continue;
					
					// Avoid duplicate visit ?
					visitedCallee.insert(callee);

					// Skip indirect calls unknown to the compiler (indirect calls),
					// calls that does not contains array expression
					// calls to functions defined in another source file or library.
					if (callee == NULL ||  callee->isDeclaration() ||
						!Detector.isProfitFunc(*callee)) {
						continue;
					}
					
					// Now callee should be a function that
					// 1) Has Array Expression
					// 2) Never visited before
					// 3) Declared in the same source file
					
					// Now we inspect this Function to see
					// 1) If this function's parameter has array expression
					// 2) If yes, bind them with the source in the caller.
					// 3) Find the private array
					
					
					workingList.push(callee);
				}
				
				// The callee receives the function call
				if (!profitCallee.empty()) {
					
				}
			}
			return true;
		}
	};
}

char ArrayEquivalenceAnalysis::ID = 0;
static RegisterPass<ArrayEquivalenceAnalysis> X("array-eq", "Analysis Array Equivalence Pass");





