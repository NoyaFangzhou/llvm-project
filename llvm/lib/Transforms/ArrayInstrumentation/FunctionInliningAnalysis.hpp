//
//  FunctionInliningAnalysis.hpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 4/22/21.
//

#ifndef FunctionInliningAnalysis_hpp
#define FunctionInliningAnalysis_hpp

#include <stdio.h>
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <map>
#include <vector>

using namespace llvm;

/* This is a function pass that inline functions with given regex */

namespace FunctionInlineAnalysis {
	struct FunctionInlineAnalysis : public FunctionPass {
		static char ID;
		FunctionInlineAnalysis();
		
		DenseSet<StringRef> functions;
		
		/* Match str with a list of regex */
		bool doesStringMatchAnyRegex(StringRef Str,
									 const cl::list<std::string> &RegexList);
		
		/* Analysis pass main function */
		bool runOnFunction(Function &F) override;
	};
}

#endif /* FunctionInliningAnalysis_hpp */
