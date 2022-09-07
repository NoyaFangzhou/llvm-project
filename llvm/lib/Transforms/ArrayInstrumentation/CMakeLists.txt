if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library(LLVMArrayInstrumentation MODULE BUILDTREE_ONLY
    # List your source files here.
    ArrayInstrumentation.cpp
    ArrayOMPInstrumentation.cpp

    ArrayInstrumentationTest.cpp
    
    ArrayIDGenerator.hpp
    ArrayIDGenerator.cpp

    ArrayInstrumentationUtils.hpp
    ArrayInstrumentationUtils.cpp
    
    ArrayEquivalenceAnalysis.hpp
    ArrayEquivalenceAnalysis.cpp
    
    FunctionInliningAnalysis.hpp
    FunctionInliningAnalysis.cpp
    

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )
