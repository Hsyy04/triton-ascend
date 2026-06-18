func.func @_hstu_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: f32, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
  %c115 = arith.constant {ssbuffer.core_type = "VECTOR"} 115 : index
  %c320 = arith.constant {ssbuffer.core_type = "VECTOR"} 320 : index
  %c320_0 = arith.constant {ssbuffer.core_type = "CUBE"} 320 : index
  %c512 = arith.constant {ssbuffer.core_type = "CUBE"} 512 : index
  %cst = arith.constant {ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16
  %c64 = arith.constant {ssbuffer.core_type = "CUBE"} 64 : index
  %c288 = arith.constant {ssbuffer.core_type = "CUBE"} 288 : index
  %c1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : index
  %c1_1 = arith.constant {ssbuffer.core_type = "CUBE"} 1 : index
  %c0 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
  %c0_2 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
  %cst_3 = arith.constant {ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
  %c64_i64 = arith.constant {MixUse, ssbuffer.core_type = "VECTOR"} 64 : i64
  %c64_i64_4 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 64 : i64
  %c0_i64 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 0 : i64
  %c0_i64_5 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 0 : i64
  %c64_i32 = arith.constant {Undefined, ssbuffer.core_type = "CUBE"} 64 : i32
  %c4_i64 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 4 : i64
  %c4_i64_6 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 4 : i64
  %c3_i32 = arith.constant {Undefined, ssbuffer.core_type = "VECTOR"} 3 : i32
  %c1_i32 = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i32
  %c1_i64 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 1 : i64
  %c1_i64_7 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 1 : i64
  %c72_i64 = arith.constant {ssbuffer.core_type = "CUBE"} 72 : i64
  %c288_i64 = arith.constant {ssbuffer.core_type = "CUBE"} 288 : i64
  %c80_i64 = arith.constant {MixUse, ssbuffer.core_type = "VECTOR"} 80 : i64
  %c80_i64_8 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 80 : i64
  %c320_i64 = arith.constant {MixUse, ssbuffer.core_type = "VECTOR"} 320 : i64
  %c320_i64_9 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 320 : i64
  %c512_i64 = arith.constant {ssbuffer.core_type = "VECTOR"} 512 : i64
  %c512_i64_10 = arith.constant {ssbuffer.core_type = "CUBE"} 512 : i64
  %c512_i32 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 512 : i32
  %c512_i32_11 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 512 : i32
  %c511_i32 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 511 : i32
  %c511_i32_12 = arith.constant {MixUse, ssbuffer.core_type = "CUBE"} 511 : i32
  %c5_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 5 : i32
  %c0_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
  %c2_i32 = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : i32
  %c511_i64 = arith.constant {Undefined, ssbuffer.core_type = "VECTOR"} 511 : i64
  %cst_13 = arith.constant {ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
  %cst_14 = arith.constant {ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
  %0 = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
  %1 = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<64x512xf32>
  %2 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_13 : f32) outs(%0 : tensor<64x512xf32>) -> tensor<64x512xf32>
  %3 = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst_14 : f32) outs(%1 : tensor<64x512xf32>) -> tensor<64x512xf32>
  %4 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_3 : f32) outs(%0 : tensor<64x512xf32>) -> tensor<64x512xf32>
  %5 = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<64x80xf32>
  %6 = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst_14 : f32) outs(%5 : tensor<64x80xf32>) -> tensor<64x80xf32>
  %7 = arith.cmpi sle, %arg12, %c64_i32 {Undefined, ssbuffer.core_type = "CUBE"} : i32
  %8 = scf.if %7 -> (i64) {
    scf.yield {Undefined, ssbuffer.core_type = "CUBE"} %c4_i64_6 : i64
  } else {
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [4], sizes: [1], strides: [1] {ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: 4>>
    %44 = memref.load %reinterpret_cast[%c0] {ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: 4>>
    scf.yield {Undefined, ssbuffer.core_type = "CUBE"} %44 : i64
  } {MixUse, ssbuffer.core_type = "CUBE"}
  %9 = arith.muli %8, %c4_i64 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %10 = arith.muli %8, %c4_i64_6 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %11 = arith.extsi %arg15 {MixUse, ssbuffer.core_type = "CUBE"} : i32 to i64
  %12 = arith.extsi %arg15 {MixUse, ssbuffer.core_type = "CUBE"} : i32 to i64
  %13 = arith.minsi %11, %9 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %14 = arith.minsi %12, %10 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %15 = arith.divsi %9, %13 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %16 = arith.divsi %10, %14 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %17 = arith.addi %15, %c1_i64 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %18 = arith.addi %16, %c1_i64_7 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %19 = arith.remsi %9, %13 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %20 = arith.remsi %10, %14 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %21 = arith.extsi %arg18 {MixUse, ssbuffer.core_type = "CUBE"} : i32 to i64
  %22 = arith.extsi %arg18 {MixUse, ssbuffer.core_type = "CUBE"} : i32 to i64
  %23 = arith.cmpi slt, %21, %13 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %24 = arith.cmpi slt, %22, %14 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %25 = arith.cmpi slt, %21, %19 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %26 = arith.cmpi slt, %22, %20 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %27 = arith.muli %21, %17 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %28 = arith.muli %22, %18 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %29 = arith.muli %19, %17 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %30 = arith.muli %20, %18 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %31 = arith.subi %21, %19 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %32 = arith.subi %22, %20 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %33 = arith.muli %31, %15 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %34 = arith.muli %32, %16 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %35 = arith.addi %29, %33 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %36 = arith.addi %30, %34 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %37 = arith.select %25, %27, %35 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %38 = arith.select %26, %28, %36 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %39 = arith.select %23, %37, %c0_i64 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %40 = arith.select %24, %38, %c0_i64_5 {MixUse, ssbuffer.core_type = "CUBE"} : i64
  %41 = arith.select %25, %17, %15 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
  %42 = arith.select %23, %41, %c0_i64 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
  %43 = arith.cmpi sge, %21, %13 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
  scf.if %43 {
  } else {
    %44 = arith.cmpi sle, %42, %c0_i64 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
    scf.if %44 {
    } else {
      %45 = arith.addi %arg13, %c511_i32 {MixUse, ssbuffer.core_type = "CUBE"} : i32
      %46 = arith.addi %arg13, %c511_i32_12 {MixUse, ssbuffer.core_type = "CUBE"} : i32
      %47 = arith.divsi %45, %c512_i32 {MixUse, ssbuffer.core_type = "CUBE"} : i32
      %48 = arith.divsi %46, %c512_i32_11 {MixUse, ssbuffer.core_type = "CUBE"} : i32
      %49 = arith.extsi %47 {MixUse, ssbuffer.core_type = "CUBE"} : i32 to i64
      %50 = arith.extsi %48 {MixUse, ssbuffer.core_type = "CUBE"} : i32 to i64
      %51 = arith.muli %42, %49 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
      %52 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%arg10 : f32) outs(%0 : tensor<64x512xf32>) -> tensor<64x512xf32>
      %53 = linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%arg11 : f32) outs(%0 : tensor<64x512xf32>) -> tensor<64x512xf32>
      scf.for %arg21 = %c0_i64_5 to %51 step %c1_i64_7  : i64 {
        %54 = arith.divsi %arg21, %49 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %55 = arith.divsi %arg21, %50 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %56 = arith.addi %39, %54 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %57 = arith.addi %40, %55 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %58 = arith.remsi %arg21, %49 {ssbuffer.core_type = "VECTOR"} : i64
        %59 = arith.remsi %arg21, %50 {ssbuffer.core_type = "CUBE"} : i64
        %60 = arith.divsi %56, %8 {MixUse, ssbuffer.core_type = "VECTOR"} : i64
        %61 = arith.divsi %57, %8 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %62 = arith.remsi %56, %8 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %63:2 = scf.if %7 -> (i64, i64) {
          scf.yield {Undefined, ssbuffer.core_type = "VECTOR, CUBE"} %62, %c0_i64_5 : i64, i64
        } else {
          %146:2 = scf.for %arg22 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg23 = %c0_i32, %arg24 = %c5_i32) -> (i32, i32)  : i32 {
            %152 = arith.addi %arg23, %arg24 {ssbuffer.core_type = "VECTOR"} : i32
            %153 = arith.divsi %152, %c2_i32 {ssbuffer.core_type = "VECTOR"} : i32
            %154 = arith.index_cast %153 {ssbuffer.core_type = "VECTOR"} : i32 to index
            %reinterpret_cast_35 = memref.reinterpret_cast %arg7 to offset: [%154], sizes: [1], strides: [1] {ssbuffer.core_type = "VECTOR"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %155 = memref.load %reinterpret_cast_35[%c0] {ssbuffer.core_type = "VECTOR"} : memref<1xi64, strided<[1], offset: ?>>
            %156 = arith.cmpi sle, %155, %62 {ssbuffer.core_type = "VECTOR"} : i64
            %157 = arith.select %156, %arg24, %153 {ssbuffer.core_type = "VECTOR"} : i32
            %158 = scf.if %156 -> (i32) {
              %159 = arith.addi %153, %c1_i32 {ssbuffer.core_type = "VECTOR"} : i32
              scf.yield {Undefined, ssbuffer.core_type = "VECTOR"} %159 : i32
            } else {
              scf.yield {Undefined, ssbuffer.core_type = "VECTOR"} %arg23 : i32
            } {ssbuffer.core_type = "VECTOR"}
            scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"} %158, %157 : i32, i32
          } {ssbuffer.core_type = "VECTOR, VECTOR"}
          %147 = arith.subi %146#0, %c1_i32 {ssbuffer.core_type = "CUBE"} : i32
          %148 = arith.extsi %147 {ssbuffer.core_type = "VECTOR"} : i32 to i64
          %149 = arith.index_cast %147 {ssbuffer.core_type = "CUBE"} : i32 to index
          %reinterpret_cast_34 = memref.reinterpret_cast %arg7 to offset: [%149], sizes: [1], strides: [1] {ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %150 = memref.load %reinterpret_cast_34[%c0] {ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
          %151 = arith.subi %62, %150 {MixUse, ssbuffer.core_type = "CUBE"} : i64
          scf.yield {Undefined, ssbuffer.core_type = "VECTOR, CUBE"} %148, %151 : i64, i64
        } {MixUse, ssbuffer.core_type = "VECTOR, CUBE"}
        %64 = arith.index_cast %63#0 {ssbuffer.core_type = "VECTOR"} : i64 to index
        %65 = arith.index_cast %63#0 {ssbuffer.core_type = "CUBE"} : i64 to index
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%64], sizes: [1], strides: [1] {ssbuffer.core_type = "VECTOR"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %reinterpret_cast_15 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [1], strides: [1] {ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %66 = memref.load %reinterpret_cast[%c0] {ssbuffer.core_type = "VECTOR"} : memref<1xi64, strided<[1], offset: ?>>
        %67 = memref.load %reinterpret_cast_15[%c0_2] {ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
        %68 = arith.addi %64, %c1 {ssbuffer.core_type = "VECTOR"} : index
        %69 = arith.addi %65, %c1_1 {ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_16 = memref.reinterpret_cast %arg5 to offset: [%68], sizes: [1], strides: [1] {ssbuffer.core_type = "VECTOR"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %reinterpret_cast_17 = memref.reinterpret_cast %arg5 to offset: [%69], sizes: [1], strides: [1] {ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %70 = memref.load %reinterpret_cast_16[%c0] {ssbuffer.core_type = "VECTOR"} : memref<1xi64, strided<[1], offset: ?>>
        %71 = memref.load %reinterpret_cast_17[%c0_2] {ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
        %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [%64], sizes: [1], strides: [1] {ssbuffer.core_type = "VECTOR"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %reinterpret_cast_19 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [1], strides: [1] {ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %72 = memref.load %reinterpret_cast_18[%c0] {ssbuffer.core_type = "VECTOR"} : memref<1xi64, strided<[1], offset: ?>>
        %73 = memref.load %reinterpret_cast_19[%c0_2] {ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
        %reinterpret_cast_20 = memref.reinterpret_cast %arg6 to offset: [%68], sizes: [1], strides: [1] {ssbuffer.core_type = "VECTOR"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %reinterpret_cast_21 = memref.reinterpret_cast %arg6 to offset: [%69], sizes: [1], strides: [1] {ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %74 = memref.load %reinterpret_cast_20[%c0] {ssbuffer.core_type = "VECTOR"} : memref<1xi64, strided<[1], offset: ?>>
        %75 = memref.load %reinterpret_cast_21[%c0_2] {ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
        %76 = arith.subi %70, %66 {ssbuffer.core_type = "VECTOR"} : i64
        %77 = arith.subi %71, %67 {ssbuffer.core_type = "CUBE"} : i64
        %78 = arith.subi %74, %72 {ssbuffer.core_type = "VECTOR"} : i64
        %79 = arith.subi %75, %73 {ssbuffer.core_type = "CUBE"} : i64
        %80 = arith.muli %61, %c72_i64 {ssbuffer.core_type = "CUBE"} : i64
        %81 = arith.muli %67, %c288_i64 {ssbuffer.core_type = "CUBE"} : i64
        %82 = arith.addi %80, %81 {ssbuffer.core_type = "CUBE"} : i64
        %83 = arith.muli %73, %c288_i64 {ssbuffer.core_type = "CUBE"} : i64
        %84 = arith.addi %80, %83 {ssbuffer.core_type = "CUBE"} : i64
        %85 = arith.muli %60, %c80_i64 {MixUse, ssbuffer.core_type = "VECTOR"} : i64
        %86 = arith.muli %61, %c80_i64_8 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %87 = arith.muli %73, %c320_i64_9 {ssbuffer.core_type = "CUBE"} : i64
        %88 = arith.addi %86, %87 {ssbuffer.core_type = "CUBE"} : i64
        %89 = arith.muli %66, %c320_i64 {MixUse, ssbuffer.core_type = "VECTOR"} : i64
        %90 = arith.muli %67, %c320_i64_9 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %91 = arith.addi %85, %89 {MixUse, ssbuffer.core_type = "VECTOR"} : i64
        %92 = arith.addi %86, %90 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %93 = arith.addi %78, %c511_i64 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
        %94 = arith.divsi %93, %c512_i64 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
        %95 = arith.muli %59, %c512_i64_10 {ssbuffer.core_type = "CUBE"} : i64
        %96 = arith.muli %63#1, %c64_i64 {MixUse, ssbuffer.core_type = "VECTOR"} : i64
        %97 = arith.muli %63#1, %c64_i64_4 {MixUse, ssbuffer.core_type = "CUBE"} : i64
        %98 = arith.index_cast %82 {ssbuffer.core_type = "CUBE"} : i64 to index
        %99 = arith.index_cast %96 {ssbuffer.core_type = "VECTOR"} : i64 to index
        %100 = arith.index_cast %97 {ssbuffer.core_type = "CUBE"} : i64 to index
        %101 = arith.muli %100, %c288 {ssbuffer.core_type = "CUBE"} : index
        %102 = arith.addi %98, %101 {ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_22 = memref.reinterpret_cast %arg2 to offset: [%102], sizes: [64, 72], strides: [288, 1] {ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<64x72xf16, strided<[288, 1], offset: ?>>
        %alloc = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x72xf16>
        %103 = arith.addi %100, %c64 {ssbuffer.core_type = "CUBE"} : index
        %104 = arith.index_cast %76 {ssbuffer.core_type = "VECTOR"} : i64 to index
        %105 = arith.index_cast %77 {ssbuffer.core_type = "CUBE"} : i64 to index
        %106 = arith.maxsi %99, %104 {ssbuffer.core_type = "VECTOR"} : index
        %107 = arith.maxsi %100, %105 {ssbuffer.core_type = "CUBE"} : index
        %108 = arith.minsi %103, %107 {ssbuffer.core_type = "CUBE"} : index
        %109 = arith.subi %108, %100 {ssbuffer.core_type = "CUBE"} : index
        %110 = arith.cmpi slt, %109, %c64 {ssbuffer.core_type = "CUBE"} : index
        scf.if %110 {
          linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc : memref<64x72xf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast_22[0, 0] [%109, 72] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<64x72xf16, strided<[288, 1], offset: ?>> to memref<?x72xf16, strided<[288, 1], offset: ?>>
        %subview_23 = memref.subview %alloc[0, 0] [%109, 72] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<64x72xf16> to memref<?x72xf16, strided<[72, 1]>>
        memref.copy %subview, %subview_23 {ssbuffer.core_type = "CUBE"} : memref<?x72xf16, strided<[288, 1], offset: ?>> to memref<?x72xf16, strided<[72, 1]>>
        %111 = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "CUBE"} : memref<64x72xf16>
        %112 = arith.index_cast %84 {ssbuffer.core_type = "CUBE"} : i64 to index
        %113 = arith.index_cast %95 {ssbuffer.core_type = "CUBE"} : i64 to index
        %114 = arith.muli %113, %c288 {ssbuffer.core_type = "CUBE"} : index
        %115 = arith.addi %112, %114 {ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_24 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [512, 72], strides: [288, 1] {ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<512x72xf16, strided<[288, 1], offset: ?>>
        %alloc_25 = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<512x72xf16>
        %116 = arith.addi %113, %c512 {ssbuffer.core_type = "CUBE"} : index
        %117 = arith.index_cast %79 {ssbuffer.core_type = "CUBE"} : i64 to index
        %118 = arith.maxsi %113, %117 {ssbuffer.core_type = "CUBE"} : index
        %119 = arith.minsi %116, %118 {ssbuffer.core_type = "CUBE"} : index
        %120 = arith.subi %119, %113 {ssbuffer.core_type = "CUBE"} : index
        %121 = arith.cmpi slt, %120, %c512 {ssbuffer.core_type = "CUBE"} : index
        scf.if %121 {
          linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc_25 : memref<512x72xf16>)
        } {hivm.unlikely_condition}
        %subview_26 = memref.subview %reinterpret_cast_24[0, 0] [%120, 72] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<512x72xf16, strided<[288, 1], offset: ?>> to memref<?x72xf16, strided<[288, 1], offset: ?>>
        %subview_27 = memref.subview %alloc_25[0, 0] [%120, 72] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<512x72xf16> to memref<?x72xf16, strided<[72, 1]>>
        memref.copy %subview_26, %subview_27 {ssbuffer.core_type = "CUBE"} : memref<?x72xf16, strided<[288, 1], offset: ?>> to memref<?x72xf16, strided<[72, 1]>>
        %122 = bufferization.to_tensor %alloc_25 restrict writable {ssbuffer.core_type = "CUBE"} : memref<512x72xf16>
        %123 = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<72x512xf16>
        %transposed = linalg.transpose ins(%122 : tensor<512x72xf16>) outs(%123 : tensor<72x512xf16>) permutation = [1, 0]  {ssbuffer.core_type = "CUBE"}
        %124 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"} ins(%111, %transposed : tensor<64x72xf16>, tensor<72x512xf16>) outs(%3 : tensor<64x512xf32>) -> tensor<64x512xf32>
        %125 = arith.mulf %124, %52 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
        %126 = arith.subf %2, %125 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
        %127 = math.exp %126 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
        %128 = arith.addf %127, %4 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
        %129 = arith.divf %4, %128 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
        %130 = arith.mulf %125, %129 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
        %131 = arith.mulf %130, %53 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32>
        %132 = arith.truncf %131 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<64x512xf32> to tensor<64x512xf16>
        %133 = arith.index_cast %88 {ssbuffer.core_type = "CUBE"} : i64 to index
        %134 = arith.muli %113, %c320_0 {ssbuffer.core_type = "CUBE"} : index
        %135 = arith.addi %133, %134 {ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_28 = memref.reinterpret_cast %arg4 to offset: [%135], sizes: [512, 80], strides: [320, 1] {ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<512x80xf16, strided<[320, 1], offset: ?>>
        %alloc_29 = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<512x80xf16>

        scf.if %121 {
          linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc_29 : memref<512x80xf16>)
        } {hivm.unlikely_condition}
        
        %subview_30 = memref.subview %reinterpret_cast_28[0, 0] [%120, 80] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<512x80xf16, strided<[320, 1], offset: ?>> to memref<?x80xf16, strided<[320, 1], offset: ?>>
        %subview_31 = memref.subview %alloc_29[0, 0] [%120, 80] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<512x80xf16> to memref<?x80xf16, strided<[80, 1]>>
        memref.copy %subview_30, %subview_31 {ssbuffer.core_type = "CUBE"} : memref<?x80xf16, strided<[320, 1], offset: ?>> to memref<?x80xf16, strided<[80, 1]>>
        %136 = bufferization.to_tensor %alloc_29 restrict writable {ssbuffer.core_type = "CUBE"} : memref<512x80xf16>
        %137 = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE", check_this} ins(%132, %136 : tensor<64x512xf16>, tensor<512x80xf16>) outs(%6 : tensor<64x80xf32>) -> tensor<64x80xf32>
        %138 = arith.index_cast %91 {ssbuffer.core_type = "VECTOR"} : i64 to index
        %139 = arith.index_cast %92 {ssbuffer.core_type = "CUBE"} : i64 to index
        %140 = arith.muli %99, %c320 {ssbuffer.core_type = "VECTOR"} : index
        %141 = arith.muli %100, %c320_0 {ssbuffer.core_type = "CUBE"} : index
        %142 = arith.addi %138, %140 {ssbuffer.core_type = "VECTOR"} : index
        %143 = arith.addi %139, %141 {ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_32 = memref.reinterpret_cast %arg8 to offset: [%143], sizes: [64, 80], strides: [320, 1] {ssbuffer.core_type = "CUBE"} : memref<?xf32> to memref<64x80xf32, strided<[320, 1], offset: ?>>
        %subview_33 = memref.subview %reinterpret_cast_32[0, 0] [%109, 80] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<64x80xf32, strided<[320, 1], offset: ?>> to memref<?x80xf32, strided<[320, 1], offset: ?>>
        %extracted_slice = tensor.extract_slice %137[0, 0] [%109, 80] [1, 1] {ssbuffer.core_type = "CUBE"} : tensor<64x80xf32> to tensor<?x80xf32>
        
        hivm.hir.store ins(%extracted_slice : tensor<?x80xf32>) outs(%subview_33 : memref<?x80xf32, strided<[320, 1], offset: ?>>) {ssbuffer.core_type = "CUBE"} atomic = <add>
        
        %144 = arith.subi %94, %c1_i64 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
        %145 = arith.cmpi eq, %58, %144 {Undefined, ssbuffer.core_type = "VECTOR"} : i64
        scf.if %145 {
          %reinterpret_cast_34 = memref.reinterpret_cast %arg8 to offset: [%142], sizes: [115, 80], strides: [320, 1] {ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<115x80xf32, strided<[320, 1], offset: ?>>
          %alloc_35 = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<115x80xf32>
          %146 = arith.addi %99, %c115 {ssbuffer.core_type = "VECTOR"} : index
          %147 = arith.minsi %146, %106 {ssbuffer.core_type = "VECTOR"} : index
          %148 = arith.subi %147, %99 {ssbuffer.core_type = "VECTOR"} : index
          %149 = arith.cmpi slt, %148, %c115 {ssbuffer.core_type = "VECTOR"} : index
          scf.if %149 {
            linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst_13 : f32) outs(%alloc_35 : memref<115x80xf32>)
          } {hivm.unlikely_condition}
          %subview_36 = memref.subview %reinterpret_cast_34[0, 0] [%148, 80] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<115x80xf32, strided<[320, 1], offset: ?>> to memref<?x80xf32, strided<[320, 1], offset: ?>>
          %subview_37 = memref.subview %alloc_35[0, 0] [%148, 80] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<115x80xf32> to memref<?x80xf32, strided<[80, 1]>>
          memref.copy %subview_36, %subview_37 {ssbuffer.core_type = "VECTOR"} : memref<?x80xf32, strided<[320, 1], offset: ?>> to memref<?x80xf32, strided<[80, 1]>>
          %150 = bufferization.to_tensor %alloc_35 restrict writable {ssbuffer.core_type = "VECTOR"} : memref<115x80xf32>
          %151 = arith.truncf %150 {DataUse, ssbuffer.core_type = "VECTOR"} : tensor<115x80xf32> to tensor<115x80xf16>
          %reinterpret_cast_38 = memref.reinterpret_cast %arg9 to offset: [%142], sizes: [115, 80], strides: [320, 1] {ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<115x80xf16, strided<[320, 1], offset: ?>>
          %extracted_slice_39 = tensor.extract_slice %151[0, 0] [%148, 80] [1, 1] {ssbuffer.core_type = "VECTOR"} : tensor<115x80xf16> to tensor<?x80xf16>
          %subview_40 = memref.subview %reinterpret_cast_38[0, 0] [%148, 80] [1, 1] {ssbuffer.core_type = "VECTOR"} : memref<115x80xf16, strided<[320, 1], offset: ?>> to memref<?x80xf16, strided<[320, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice_39 in writable %subview_40 {ssbuffer.core_type = "VECTOR"} : (tensor<?x80xf16>, memref<?x80xf16, strided<[320, 1], offset: ?>>) -> ()
        } {Undefined}
      } {Undefined}
    } {Undefined}
  } {Undefined}
  return
}