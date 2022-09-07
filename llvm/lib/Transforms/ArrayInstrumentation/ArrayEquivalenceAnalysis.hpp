//
//  ArrayEquivalenceAnalysis.hpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 4/20/21.
//

#ifndef ArrayEquivalenceAnalysis_hpp
#define ArrayEquivalenceAnalysis_hpp

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include"llvm/Analysis/CallGraph.h"

using namespace llvm;


class ArrayFuncNode {
	TinyPtrVector<Value> Shared;
	TinyPtrVector<Value> Private;
public:
	ArrayFuncNode();
	bool isPrivate(Value * array);
	bool isShared(Value * array);
};


class ArrayEqDetector {
private:
	
public:
	ArrayEqDetector();
	bool isProfitFunc(Function &F);
	
	
	
};


#endif /* ArrayEquivalenceAnalysis_hpp */
