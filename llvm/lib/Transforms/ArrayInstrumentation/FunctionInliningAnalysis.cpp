//
//  FunctionInliningAnalysis.cpp
//  LLVMArrayInstrumentation
//
//  Created by noya-fangzhou on 4/22/21.
//
#include "ArrayInstrumentationUtils.hpp"
#include "FunctionInliningAnalysis.hpp"

#define DEBUG_TYPE "funciton-inline"

using namespace llvm;

static cl::list<std::string> InstrumentedFunctions(
	"func",
	cl::desc("Functions that match a regex. "
			 "Multiple regexes can be comma separated. "
			 "functions that match ANY of the regexes provided"
			 "will be instrumented."),
	cl::ZeroOrMore, cl::CommaSeparated);

namespace FunctionInlineAnalysis {

	char FunctionInlineAnalysis::ID = 0;
	static RegisterPass<FunctionInlineAnalysis> X("-ilfunc", "Inline function calls for specific functions");

	FunctionInlineAnalysis::FunctionInlineAnalysis() : FunctionPass(ID) {}
		
		
	/// Check if a string matches any regex in a list of regexes.
	/// @param Str the input string to match against.
	/// @param RegexList a list of strings that are regular expressions.
	bool FunctionInlineAnalysis::doesStringMatchAnyRegex(StringRef Str,
								const cl::list<std::string> &RegexList) {
		for (auto RegexStr : RegexList) {
			Regex R(RegexStr);

			std::string Err;
			if (!R.isValid(Err))
			report_fatal_error("invalid regex given as input to polly: " + Err, true);

			if (R.match(Str))
			return true;
		}
		return false;
	}

	/// Call for every function
	bool FunctionInlineAnalysis::runOnFunction(Function &F) {
		bool beginAnalysis = doesStringMatchAnyRegex(F.getName(), InstrumentedFunctions);
		LLVM_DEBUG(dbgs() << "Function " << F.getName() << " " << beginAnalysis << "\n");
		if (beginAnalysis) {
			functions.insert(F.getName());
			for (auto & B : F) {
				for (auto & I : B) {
					if (isa<CallInst>(&I)) {
						InlineFunctionInfo ifi;
						CallBase * callInst = dyn_cast<CallBase>(&I);
						InlineFunction(*callInst, ifi);
						LLVM_DEBUG(dbgs() << "Inline Function " << callInst->getCalledFunction()->getName() << " in " << F.getName());
					}
				}
			}
		}
		return true;
	}
}
