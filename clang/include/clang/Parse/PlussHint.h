//
// Created by noya-fangzhou on 10/19/21.
//

#ifndef LLVM_CLANG_PARSE_PLUSSHINT_H
#define LLVM_CLANG_PARSE_PLUSSHINT_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"

namespace clang {

/// Locality analysis hint for pluss pragmas.
struct PlussHint {
  // Source range of the directive.
  SourceRange Range;
  // Identifier corresponding to the name of the pragma.  "pluss" for
  // "#pragma pluss" directives.
  IdentifierLoc *PragmaNameLoc;
  // Name of the pluss hint.  Examples: "on", "off", "array".  In the
  // "#pragma unroll" and "#pragma nounroll" cases, this is identical to
  // PragmaNameLoc.
  IdentifierLoc *OptionLoc;
  // Identifier for the hint state argument.  If null, then the state is
  // default value such as for "#pragma unroll".
  IdentifierLoc *StateLoc;
  // Expression for the hint argument if it exists, null otherwise.
  Expr *ValueExpr;

  PlussHint()
      : PragmaNameLoc(nullptr), OptionLoc(nullptr), StateLoc(nullptr),
        ValueExpr(nullptr) {}
};

} // end namespace clang


#endif // LLVM_CLANG_PARSE_PLUSSHINT_H
