cmake_minimum_required(VERSION 3.6)
PROJECT(toy)

add_subdirectory(include)
set(LLVM_LINK_COMPONENTS Core Support )
add_compile_options(-frtti)

find_package(Boost REQUIRED COMPONENTS python-py27)
set(PYTHON_DOT_VERSION 2.7)
set(PYTHON_INCLUDE /usr/include/python2.7)
set(PYTHON_LIBRARY /usr/lib/python2.7/config-x86_64-linux-gnu)

include_directories(${PYTHON_INCLUDE} ${Boost_INCLUDE_DIRS})
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lrt -O3")
SET(LIBNAME relay)


set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ToyTECombineIncGen)


add_library(ToyDialect mlir/Dialect.cpp)
add_library(ToyShapeInferencePass mlir/ShapeInferencePass.cpp)
add_library(ToyLowerToAffineLoops mlir/LowerToAffineLoops.cpp )
add_library(ToyLowerToLLVM mlir/LowerToLLVM.cpp)
add_library(ToyCombine mlir/ToyCombine.cpp)


add_library(${LIBNAME} SHARED   main.cpp)
add_dependencies(ToyCombine ToyTECombineIncGen)
add_dependencies(ToyShapeInferencePass ToyTEShapeInferenceInterfaceIncGen)
add_dependencies(ToyDialect ToyCombine)

add_dependencies(${LIBNAME} ToyTEShapeInferenceInterfaceIncGen)
add_dependencies(${LIBNAME} ToyTEOpsIncGen)
add_dependencies(${LIBNAME} ToyTECombineIncGen)
add_dependencies(${LIBNAME} MLIRCallInterfacesIncGen)
add_dependencies(${LIBNAME} MLIRLoopToStandard)
add_dependencies(${LIBNAME} MLIRAffineToStandard)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
TARGET_LINK_LIBRARIES(${LIBNAME} ${Boost_LIBRARIES}

    
    MLIRLoopToStandard
    MLIRAffineToStandard
    MLIRStandardToLLVM
    ToyLowerToLLVM

    ${dialect_libs}
    ${conversion_libs}
    MLIRCallInterfaces
    MLIRSideEffects
    MLIRSupport    

    MLIRAnalysis
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRPass
    MLIRTargetLLVMIR
    MLIRTransforms

    ToyCombine
    ToyDialect
    ToyShapeInferencePass
    ToyLowerToAffineLoops
    
    
 -lpython2.7 -fPIC -frtti -shared)
