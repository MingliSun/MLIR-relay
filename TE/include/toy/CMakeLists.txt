set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls "-I${CMAKE_CURRENT_SOURCE_DIR}/..")
mlir_tablegen(Ops.cpp.inc -gen-op-defs "-I${CMAKE_CURRENT_SOURCE_DIR}/..")
add_public_tablegen_target(ToyTEOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls "-I${CMAKE_CURRENT_SOURCE_DIR}/..")
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs "-I${CMAKE_CURRENT_SOURCE_DIR}/..")
add_public_tablegen_target(ToyTEShapeInferenceInterfaceIncGen)
