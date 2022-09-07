#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "FunctionInliningAnalysis.hpp"
#include "ArrayIDGenerator.hpp"
#include "ArrayInstrumentationUtils.hpp"

#include <unordered_set>
#include <set>

#define DEBUG_TYPE "array-instr"

using namespace llvm;
using namespace std;

static cl::opt<bool> EnableCodeOptimization(
	"opt-enable",
	cl::desc("Open the code elimination optimization"),
	cl::init(false));
static cl::opt<bool> PerArrayAnalysis(
	"per-array",
	cl::desc("Output the locality analysis for ech array"),
	cl::init(false));

namespace {
	struct ArrayInstrumentation : public FunctionPass {
		static char ID;
		ArrayInstrumentation() : FunctionPass(ID) {}
		ArrayIDGenerator Generator;
		
		void findAllDefs(Value * v, unordered_set<Instruction *> &defs)
		{
			if (!isa<Instruction>(v)) return;
			Instruction * I = dyn_cast<Instruction>(v);
			defs.insert(I);
			for (Use &U : I->operands()) {
				Value * def = U.get();
				findAllDefs(def, defs);
			}
		}
		
		
		void findAllUses(Instruction * I, unordered_set<Instruction *> &users)
		{
			if (I->getNumUses() > 0) {
				for (User * U : I->users()) {
					if (Instruction *userInst = dyn_cast<Instruction>(U)) {
						users.insert(userInst);
						 findAllUses(userInst, users);
					}
				}
			}
		}
		
		
		int instrumentAnalysisFunc(FunctionCallee func, BasicBlock * B, Instruction * I)
		{
			IRBuilder<> builder(I);
			builder.SetInsertPoint(B, builder.GetInsertPoint());
			builder.CreateCall(func.getFunctionType(), func.getCallee());
			return 0;
		}
		
		/// Instrument a runtime function
		int instrumentMemoryAccess(Function * F, FunctionCallee func, BasicBlock * B, GetElementPtrInst * I)
		{
			// get the array index
			Value * arrayIdx = I->getOperand(1);
			// get the name of the array
			Value * arrayName = getArrayNameExpr(I->getPointerOperand());
			// right there, all irrelevant instruction can be removed
			// irrelevant instructions are those not in the def-use chain
			if (!arrayIdx || !arrayName)
				return -1;
			// Create the ArrayName obj
			ArrayName * name = new ArrayName(arrayName, F);
			LLVM_DEBUG(dbgs() << "Instrument array " << * arrayName << " "
					   << *arrayName << "\n");
			IRBuilder<> builder(I);
			// ++ here means inserting after instruction I
			builder.SetInsertPoint(B, ++builder.GetInsertPoint());
			Value * args[] = {builder.getInt32(Generator.CreateArrayID(name)), arrayIdx};
			builder.CreateCall(func.getFunctionType(), func.getCallee(), args);
			return 0;
		}
		
		void getAnalysisUsage(AnalysisUsage & AU) const override {
			AU.setPreservesAll();
			AU.addRequired<FunctionInlineAnalysis::FunctionInlineAnalysis>();
		}

		/// Call for every function
		virtual bool runOnFunction(Function &F) override {
			
			int ret;
			int removeInstCnt = 0;
			
			DenseSet<StringRef> candidates = getAnalysis<FunctionInlineAnalysis::FunctionInlineAnalysis>().functions;
			// Get the function to call from our runtime library.
			LLVMContext &Ctx = F.getContext();
			std::vector<Type*> paramTypes = {Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)};
			Type *retType = Type::getVoidTy(Ctx);
			FunctionType *logFuncType = FunctionType::get(retType, paramTypes, false);
			FunctionCallee logFunc = F.getParent()->getOrInsertFunction("rtTmpAccess", logFuncType);
			
			// Get the function to call at the end of the program
			// These functions includes:
			// 1) Func conver RT to MRC
			// 2) RT and MRC dump function
			FunctionType *analysisFuncType = FunctionType::get(retType, false);
			FunctionCallee InitFunc = F.getParent()->getOrInsertFunction("libInit", analysisFuncType);
			FunctionCallee RTToMRCFunc = F.getParent()->getOrInsertFunction("RTtoMR_AET", analysisFuncType);
			FunctionCallee RTDumpFunc = F.getParent()->getOrInsertFunction("RTDump", analysisFuncType);
			FunctionCallee PerArrayRTDumpFunc = F.getParent()->getOrInsertFunction("perArrayRTDump", analysisFuncType);
			FunctionCallee MRCDumpFunc = F.getParent()->getOrInsertFunction("MRDump", analysisFuncType);
			FunctionCallee TerminateFunc = F.getParent()->getOrInsertFunction("libTerminate", analysisFuncType);
			
			bool ismain = F.getName().contains("main");
			
			bool beginInstr = candidates.find(F.getName()) != candidates.end();
			LLVM_DEBUG(dbgs() << "Function " << F.getName() << " " << beginInstr << "\n");
			if (ismain) {
				for (auto &I: F.getEntryBlock()) {
					instrumentAnalysisFunc(InitFunc, &F.getEntryBlock(), &I);
					break;
				}
			}
			if (beginInstr || ismain) {
				// Here we store all instructions that the array idx instruction depends
				unordered_set<Instruction *> defs;
				for (auto &B : F) {
					for (auto I = B.begin(); I != B.end(); ++I) {
						Instruction & inst = *I;
						if (beginInstr && isa<GetElementPtrInst>(&inst)) {
							GetElementPtrInst * arrayInst = dyn_cast<GetElementPtrInst>(&inst);
							LLVM_DEBUG(dbgs() << "Instrument after " << *arrayInst << "\n");
							ret = instrumentMemoryAccess(&F, logFunc, &B, arrayInst);
							if (ret < 0)
								LLVM_DEBUG(dbgs() << " FAILED \n");
							else
								LLVM_DEBUG(dbgs() << " SUCCESS \n");
							
							if (EnableCodeOptimization) {
								// Here we store all instructions that use the array element
								unordered_set<Instruction *> users;
								findAllUses(arrayInst, users);
								
								// Here we traverse the set in reverse order
								// instruction with no dependence will be pop out first
								unordered_set<Instruction *>::iterator sit = users.begin();
								while (sit != users.end()) {
									LLVM_DEBUG(dbgs() << "User: " << *(*sit) << "\n");
									if (!(*sit)->user_empty())
										(*sit)->replaceAllUsesWith(UndefValue::get((*sit)->getType()));
									(*sit)->eraseFromParent();
									sit++;
									removeInstCnt += 1;
								}
								// remove the getelementptr instruction
								if (!arrayInst->user_empty()) {
									arrayInst->replaceAllUsesWith(UndefValue::get(arrayInst->getType()));
								}
								I = arrayInst->eraseFromParent();
								removeInstCnt += 1;
							}
							
						}
						/// At the end of the function, insert all function that dump the analysis result
						if (ismain && isa<ReturnInst>(&inst)) {
							ReturnInst * retInst = dyn_cast<ReturnInst>(&inst);
							instrumentAnalysisFunc(RTToMRCFunc, &B, retInst);
							instrumentAnalysisFunc(RTDumpFunc, &B, retInst);
							instrumentAnalysisFunc(MRCDumpFunc, &B, retInst);
							instrumentAnalysisFunc(TerminateFunc, &B, retInst);
						}
					}
				}
				if (ismain) {
					Generator.print();
				}
				LLVM_DEBUG(
						   if (EnableCodeOptimization) dbgs() << "Total " << removeInstCnt <<
						   " Instruction removed by the optimization \n");
			}
			return true;
		}
	};
}

char ArrayInstrumentation::ID = 0;
static RegisterPass<ArrayInstrumentation> X("array-instr", "Instrument Array Access Pass");
