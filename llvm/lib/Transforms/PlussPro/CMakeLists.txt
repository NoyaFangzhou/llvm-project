if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library(LLVMPlussAnalysis MODULE BUILDTREE_ONLY
        # List your source files here.
        LoopAnalysisWrapperPass.h
        LoopAnalysisWrapperPass.cpp
        PlussAbstractionTreeAnalysis.h
        PlussAbstractionTreeAnalysis.cpp
        BranchAnalysis.h
        BranchAnalysis.cpp
        InductionVarAnalysis.h
        InductionVarAnalysis.cpp
        AccessGraphAnalysis.h
        AccessGraphAnalysis.cpp

        ModelValidationAnalysis.h
        ModelValidationAnalysis.cpp
        SamplerCodeGen.h
        SamplerCodeGen.cpp
        # this sampler codegen enable the rs for the whole program
        SamplerCodeGenPro.h
        SamplerCodeGenPro.cpp

        # this sampler codegen enable the model optimization
        ModelCodeGen.h
        ModelCodeGen.cpp
        ModelCodeGenPro.h
        ModelCodeGenPro.cpp

        # old sps sampler codegen
        OldSamplerCodeGen.h
        OldSamplerCodeGen.cpp
        IterationCounterCodeGen.h
        IterationCounterCodeGen.cpp
        # mem profiler -- runtime solution
        MemProfilerPass.h
        MemProfilerPass.cpp

        # analyze the reference idx and group them into different types
        ReferenceAnalysis.h
        ReferenceAnalysis.cpp

        # utils
        SamplerCodeGenUtils.h
        SamplerCodeGenUtils.cpp
        AccessGraphAnalysisUtils.h
        AccessGraphAnalysisUtils.cpp
        LoopAnalysisUtils.h
        LoopAnalysisUtils.cpp
        PlussUtils.h
        PlussUtils.cpp

        # wrapper of all analysis pass
        PlussAnalysisWrapperPass.cpp
    

DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )
