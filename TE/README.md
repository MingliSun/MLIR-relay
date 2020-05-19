# 项目介绍

本项目旨在探讨TVM和MLIR融合的可能性，项目前端使用boost python将原有的c++代码封装成python API，这些API会生成MLIR的第一层的Dialect（与API本身最接近的一种API，命名Relay Dialect），随后Relay Dialect在MLIR内部Lower到Affine和Std Dialect,最后Affine和Std Dialect Lower到LLVM Dialect，最终生成LLVM代码，可以使用llvm的工具lli来直接运行llvm代码。

# build 说明



将本文件夹克隆到path/to/llvm-project/mlir/example/toy/

修改CMakeLists.txt:

在llvm-project/mlir/example/toy/CMakeLists.txt 添加

```shell
add_subdirectory(TE)
```

在llvm-project/mlir/test/CMakeLists.txt添加(只是添加`relay`)

```shell
if(LLVM_BUILD_EXAMPLES)
  list(APPEND MLIR_TEST_DEPENDS
    toyc-ch1
    toyc-ch2
    toyc-ch3
    toyc-ch4
    toyc-ch5
    toyc-ch6
    toyc-ch7
    relay
    )
endif()
```

进行build(参考 https://mlir.llvm.org/getting_started/ )

```shell
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON\ #增加了本条

cmake --build . --target check-mlir
```

# 测试说明

在生成的.so文件夹下（一般为build/lib）新建test.py,并使用python2运行即可

```python
import librelay
relay = librelay.getclass()
f = relay.Prototype(0)
x = relay.var("x",[3,3],[1,0,1,0,1,0,1,0,1])
y = relay.var("y",[5,5],[1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0])
z = relay.conv2d(x,y)
f1 = relay.Function([x,y],z)
mod = relay.Module(f1)
# relay.dumpMLIR(mod)
# print("-"*40)
# librelay.dumpMLIRAffine(mod)
# print("-"*40)
librelay.dumpMLIRLLVM(mod)
print("-"*40)
librelay.dumpLLVM(mod)
```

```shell
python2 test.py
```

