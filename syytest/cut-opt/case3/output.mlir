module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_swa_fwd_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32, %arg31: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c2_i32 = arith.constant {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} 2 : i32
    %0 = arith.muli %arg24, %arg25 {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %1 = arith.muli %arg24, %c2_i32 {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %c1 = arith.constant {ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %c128_i32 = arith.constant {ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c127_i32 = arith.constant {ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} 127 : i32
    %c128 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 128 : index
    %c1_0 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 1 : index
    %c0 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 0 : index
    %c0_i32 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %c16_i32 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 16 : i32
    %c4_i32 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 4 : i32
    %c128_i32_1 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 128 : i32
    %cst = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %3 = linalg.fill {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %c128_2 = arith.constant {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} 128 : index
    %c0_3 = arith.constant {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c16_i32_4 = arith.constant {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} 16 : i32
    %cst_5 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : bf16
    %cst_6 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0xFF800000 : f32
    %c1_i32 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c0_i32_7 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_8 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %4 = tensor.empty() {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
    %5 = linalg.fill {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_8 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
    %6 = tensor.empty() {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %7 = linalg.fill {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_8 : f32) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %8 = linalg.fill {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_6 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
    %cst_9 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} -1.000000e+06 : f32
    %9 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_9 : f32) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %10 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%arg11 : f32) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %c0_i8 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 0 : i8
    %c3_i32 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 3 : i32
    %c4_i32_10 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 4 : i32
    %c255_i32 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 255 : i32
    %11 = tensor.empty() {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
    %12 = linalg.fill {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} ins(%c0_i8 : i8) outs(%11 : tensor<128x128xi8>) -> tensor<128x128xi8>
    %13 = arith.muli %arg24, %c3_i32 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %14 = arith.subi %c0_i32_7, %arg24 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %15 = scf.for %arg32 = %c0_i32 to %arg8 step %c1_i32 iter_args(%arg33 = %c0_i32_7) -> (i32)  : i32 {
      %16 = arith.index_cast %arg32 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%16], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %17 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %18 = arith.addi %16, %c1_0 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg9 to offset: [%18], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %19 = memref.load %reinterpret_cast_11[%c0] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %reinterpret_cast_12 = memref.reinterpret_cast %arg10 to offset: [%16], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %20 = memref.load %reinterpret_cast_12[%c0] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %reinterpret_cast_13 = memref.reinterpret_cast %arg10 to offset: [%18], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %21 = memref.load %reinterpret_cast_13[%c0] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %22 = arith.subi %19, %17 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %23 = arith.subi %21, %20 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %24 = arith.muli %17, %arg17 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %25 = arith.index_cast %24 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %26 = arith.muli %20, %arg19 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %27 = arith.index_cast %26 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %28 = arith.muli %20, %arg21 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %29 = arith.index_cast %28 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %30 = arith.index_cast %arg32 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg9 to offset: [%30], sizes: [1], strides: [1] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %31 = memref.load %reinterpret_cast_14[%c0_3] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %32 = arith.addi %30, %c1 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg9 to offset: [%32], sizes: [1], strides: [1] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %33 = memref.load %reinterpret_cast_15[%c0_3] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %34 = arith.subi %33, %31 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %35 = arith.addi %34, %c127_i32 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %36 = arith.divsi %35, %c128_i32 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %37 = arith.muli %arg33, %c16_i32_4 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %38 = arith.addi %arg33, %36 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %39 = arith.muli %36, %c16_i32_4 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %40 = arith.addi %37, %arg29 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %41 = arith.remsi %40, %arg26 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %42 = arith.muli %31, %arg12 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %43 = arith.index_cast %42 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %44 = arith.muli %31, %arg14 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %45 = arith.index_cast %44 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %reinterpret_cast_16 = memref.reinterpret_cast %arg10 to offset: [%30], sizes: [1], strides: [1] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %46 = memref.load %reinterpret_cast_16[%c0_3] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %reinterpret_cast_17 = memref.reinterpret_cast %arg10 to offset: [%32], sizes: [1], strides: [1] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %47 = memref.load %reinterpret_cast_17[%c0_3] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %48 = arith.subi %47, %46 {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %49 = arith.subi %48, %34 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : i32
      scf.for %arg34 = %41 to %39 step %arg26  : i32 {
        %50 = arith.divsi %arg34, %c16_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %51 = arith.remsi %arg34, %c16_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %52 = arith.remsi %51, %c4_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %53 = arith.muli %50, %c128_i32_1 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %54 = arith.muli %51, %arg18 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %55 = arith.index_cast %54 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %56 = arith.addi %25, %55 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %57 = arith.maxsi %53, %c0_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %58 = arith.index_cast %57 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %59 = arith.index_cast %arg17 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %60 = arith.muli %58, %59 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %61 = arith.addi %60, %56 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %62 = arith.index_cast %22 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %reinterpret_cast_18 = memref.reinterpret_cast %arg5 to offset: [%61], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %63 = arith.divsi %60, %59 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %64 = arith.subi %62, %63 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %65 = arith.maxsi %64, %c0 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %66 = arith.minsi %65, %c128 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %67 = arith.remsi %60, %59 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %68 = arith.subi %c128, %67 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %69 = arith.maxsi %68, %c0 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %70 = arith.minsi %69, %c128 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %71 = arith.subi %c0_i32, %53 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %72 = arith.maxsi %71, %c0_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %73 = arith.index_cast %72 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %74 = arith.minsi %73, %66 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %75 = arith.subi %66, %74 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %76 = arith.minsi %70, %c0 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %77 = arith.subi %70, %76 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %78 = arith.cmpi slt, %75, %c128 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %79 = arith.cmpi slt, %77, %c128 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %80 = arith.ori %78, %79 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i1
        %subview = memref.subview %reinterpret_cast_18[0, 0] [%75, %77] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %81 = arith.muli %52, %arg20 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %82 = arith.index_cast %81 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %83 = arith.addi %27, %82 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %84 = arith.muli %52, %arg22 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %85 = arith.index_cast %84 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %86 = arith.addi %29, %85 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %alloc = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %subview_19 = memref.subview %alloc[%74, %76] [%75, %77] [1, 1] {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
        scf.if %80 {
          linalg.fill {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_5 : bf16) outs(%alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 17 : i32}
        memref.copy %subview, %subview_19 {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
        %87 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %88 = arith.divsi %arg34, %c16_i32_4 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %89 = arith.muli %88, %c128_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %90 = arith.subi %89, %34 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %91 = arith.maxsi %90, %14 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %92 = arith.minsi %91, %c0_i32_7 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %93 = arith.index_cast %0 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %94 = arith.index_cast %1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %95 = arith.addi %93, %94 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : index
        %96 = arith.index_cast %92 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %97 = arith.index_cast %arg25 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %98 = arith.muli %96, %97 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : index
        %99 = arith.addi %95, %98 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg23 to offset: [%99], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
        %alloc_21 = memref.alloc() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
        memref.copy %reinterpret_cast_20, %alloc_21 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
        %100 = bufferization.to_tensor %alloc_21 restrict writable {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
        %101 = arith.addi %89, %c128_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %102 = arith.minsi %101, %34 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %103 = arith.subi %102, %89 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %104 = arith.addi %89, %49 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %105 = arith.addi %104, %103 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %106 = arith.addi %105, %c127_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %107 = arith.divsi %106, %c128_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %108 = arith.minsi %107, %c1_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %109 = arith.subi %104, %c255_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %110 = arith.maxsi %109, %c0_i32_7 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %111 = arith.divsi %110, %c128_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %112 = arith.maxsi %108, %111 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %113:3 = scf.for %arg35 = %c0_i32 to %108 step %c1_i32 iter_args(%arg36 = %8, %arg37 = %5, %arg38 = %7) -> (tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>)  : i32 {
          %173 = arith.muli %arg35, %c128_i32_1 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %174 = arith.maxsi %173, %c0_i32 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %175 = arith.index_cast %174 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %176 = arith.index_cast %arg19 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %177 = arith.muli %175, %176 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %178 = arith.index_cast %23 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %179 = arith.index_cast %arg21 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %180 = arith.muli %175, %179 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %181 = arith.divsi %177, %176 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %182 = arith.subi %178, %181 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %183 = arith.maxsi %182, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %184 = arith.minsi %183, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %185 = arith.remsi %177, %176 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %186 = arith.subi %c128, %185 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %187 = arith.maxsi %186, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %188 = arith.minsi %187, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %189 = arith.subi %c0_i32, %173 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %190 = arith.maxsi %189, %c0_i32 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %191 = arith.index_cast %190 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %192 = arith.minsi %191, %184 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %193 = arith.subi %184, %192 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %194 = arith.minsi %188, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %195 = arith.subi %188, %194 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %196 = arith.cmpi slt, %193, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %197 = arith.cmpi slt, %195, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %198 = arith.ori %196, %197 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i1
          %199 = arith.divsi %180, %179 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %200 = arith.subi %178, %199 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %201 = arith.maxsi %200, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %202 = arith.minsi %201, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %203 = arith.remsi %180, %179 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %204 = arith.subi %c128, %203 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %205 = arith.maxsi %204, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %206 = arith.minsi %205, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %207 = arith.minsi %191, %202 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %208 = arith.subi %202, %207 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %209 = arith.minsi %206, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %210 = arith.subi %206, %209 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %211 = arith.cmpi slt, %208, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %212 = arith.cmpi slt, %210, %c128 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %213 = arith.ori %211, %212 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i1
          %alloc_30 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %198 {
            linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_5 : bf16) outs(%alloc_30 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
          %214 = arith.addi %177, %83 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_31 = memref.reinterpret_cast %arg6 to offset: [%214], sizes: [128, 128], strides: [%176, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_32 = memref.subview %reinterpret_cast_31[0, 0] [%193, %195] [1, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_33 = memref.subview %alloc_30[%192, %194] [%193, %195] [1, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_32, %subview_33 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %215 = bufferization.to_tensor %alloc_30 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %216 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xbf16>
          %transposed = linalg.transpose ins(%215 : tensor<128x128xbf16>) outs(%216 : tensor<128x128xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"}
          %217 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%87, %transposed : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
          
          %218 = arith.muli %arg35, %c128_i32 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %219 = arith.subi %218, %48 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %220 = arith.maxsi %219, %14 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %221 = arith.minsi %220, %c0_i32_7 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %222 = arith.index_cast %arg24 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %223 = arith.addi %93, %222 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %224 = arith.index_cast %221 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %225 = arith.addi %223, %224 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_34 = memref.reinterpret_cast %arg23 to offset: [%225], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_35 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_34, %alloc_35 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %226 = bufferization.to_tensor %alloc_35 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %227 = arith.subi %218, %c4_i32_10 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %228 = arith.maxsi %227, %14 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %229 = arith.minsi %228, %c0_i32_7 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %230 = arith.index_cast %229 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %231 = arith.addi %223, %230 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_36 = memref.reinterpret_cast %arg23 to offset: [%231], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_37 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_36, %alloc_37 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %232 = bufferization.to_tensor %alloc_37 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %233 = arith.addi %218, %c255_i32 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %234 = arith.subi %233, %104 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %235 = arith.maxsi %234, %14 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %236 = arith.minsi %235, %arg24 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %237 = arith.index_cast %236 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %238 = arith.addi %222, %237 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_38 = memref.reinterpret_cast %arg23 to offset: [%238], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_39 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_38, %alloc_39 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %239 = bufferization.to_tensor %alloc_39 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %240 = arith.ori %232, %239 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %241 = arith.subi %218, %104 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %242 = arith.maxsi %241, %14 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %243 = arith.minsi %242, %arg24 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %244 = arith.index_cast %13 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %245 = arith.index_cast %243 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %246 = arith.addi %244, %245 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_40 = memref.reinterpret_cast %arg23 to offset: [%246], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_41 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_40, %alloc_41 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %247 = bufferization.to_tensor %alloc_41 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %248 = arith.andi %240, %247 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %249 = arith.andi %248, %100 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %250 = arith.andi %249, %226 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          
          %251 = arith.mulf %217, %10 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %252 = arith.cmpi ne, %250, %12 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %253 = arith.select %252, %251, %9 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi1>, tensor<128x128xf32>
          %reduced = linalg.reduce ins(%253 : tensor<128x128xf32>) outs(%8 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %269 = arith.maximumf %in, %init : f32
              linalg.yield %269 : f32
            }
          %254 = arith.maximumf %arg36, %reduced {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_42 = linalg.broadcast ins(%254 : tensor<128xf32>) outs(%6 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"}
          %255 = arith.subf %253, %broadcasted_42 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %256 = math.exp %255 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %257 = arith.truncf %256 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>
          %reduced_43 = linalg.reduce ins(%256 : tensor<128x128xf32>) outs(%5 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %269 = arith.addf %in, %init {ssbuffer.block_id = 21 : i32} : f32
              linalg.yield %269 {ssbuffer.block_id = 21 : i32} : f32
            }
          %258 = arith.subf %arg36, %254 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %259 = math.exp %258 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %260 = arith.mulf %arg37, %259 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %261 = arith.addf %260, %reduced_43 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_44 = linalg.broadcast ins(%259 : tensor<128xf32>) outs(%6 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"}
          %alloc_45 = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %213 {
            linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_5 : bf16) outs(%alloc_45 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
          %262 = arith.addi %180, %86 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_46 = memref.reinterpret_cast %arg7 to offset: [%262], sizes: [128, 128], strides: [%179, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_47 = memref.subview %reinterpret_cast_46[0, 0] [%208, %210] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_48 = memref.subview %alloc_45[%207, %209] [%208, %210] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_47, %subview_48 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %263 = bufferization.to_tensor %alloc_45 restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %264 = tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
          %265 = linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%264 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %266 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%257, %263 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%265 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %267 = arith.mulf %arg38, %broadcasted_44 {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %268 = arith.addf %266, %267 {ssbuffer.add_from_matmul, ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"} %254, %261, %268 : tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>
        } {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"}
        %114:3 = scf.for %arg35 = %112 to %107 step %c1_i32 iter_args(%arg36 = %113#0, %arg37 = %113#1, %arg38 = %113#2) -> (tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>)  : i32 {
          %173 = arith.muli %arg35, %c128_i32_1 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %174 = arith.maxsi %173, %c0_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %175 = arith.index_cast %174 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %176 = arith.index_cast %arg19 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %177 = arith.muli %175, %176 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %178 = arith.index_cast %23 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %179 = arith.index_cast %arg21 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %180 = arith.muli %175, %179 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %181 = arith.divsi %177, %176 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %182 = arith.subi %178, %181 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %183 = arith.maxsi %182, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %184 = arith.minsi %183, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %185 = arith.remsi %177, %176 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %186 = arith.subi %c128, %185 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %187 = arith.maxsi %186, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %188 = arith.minsi %187, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %189 = arith.subi %c0_i32, %173 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %190 = arith.maxsi %189, %c0_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %191 = arith.index_cast %190 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %192 = arith.minsi %191, %184 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %193 = arith.subi %184, %192 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %194 = arith.minsi %188, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %195 = arith.subi %188, %194 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %196 = arith.cmpi slt, %193, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %197 = arith.cmpi slt, %195, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %198 = arith.ori %196, %197 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i1
          %199 = arith.divsi %180, %179 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %200 = arith.subi %178, %199 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %201 = arith.maxsi %200, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %202 = arith.minsi %201, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %203 = arith.remsi %180, %179 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %204 = arith.subi %c128, %203 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %205 = arith.maxsi %204, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %206 = arith.minsi %205, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %207 = arith.minsi %191, %202 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %208 = arith.subi %202, %207 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %209 = arith.minsi %206, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %210 = arith.subi %206, %209 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %211 = arith.cmpi slt, %208, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %212 = arith.cmpi slt, %210, %c128 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %213 = arith.ori %211, %212 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i1
          %alloc_30 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %198 {
            linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_5 : bf16) outs(%alloc_30 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 12 : i32}
          %214 = arith.addi %177, %83 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_31 = memref.reinterpret_cast %arg6 to offset: [%214], sizes: [128, 128], strides: [%176, 1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_32 = memref.subview %reinterpret_cast_31[0, 0] [%193, %195] [1, 1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_33 = memref.subview %alloc_30[%192, %194] [%193, %195] [1, 1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_32, %subview_33 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %215 = bufferization.to_tensor %alloc_30 restrict writable {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %216 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xbf16>
          %transposed = linalg.transpose ins(%215 : tensor<128x128xbf16>) outs(%216 : tensor<128x128xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"}
          %217 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%87, %transposed : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %218 = arith.muli %arg35, %c128_i32 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %219 = arith.subi %218, %48 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %220 = arith.maxsi %219, %14 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %221 = arith.minsi %220, %c0_i32_7 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %222 = arith.index_cast %arg24 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %223 = arith.addi %93, %222 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %224 = arith.index_cast %221 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %225 = arith.addi %223, %224 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_34 = memref.reinterpret_cast %arg23 to offset: [%225], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_35 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_34, %alloc_35 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %226 = bufferization.to_tensor %alloc_35 restrict writable {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %227 = arith.subi %218, %104 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %228 = arith.maxsi %227, %14 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %229 = arith.minsi %228, %arg24 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %230 = arith.index_cast %13 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %231 = arith.index_cast %229 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %232 = arith.addi %230, %231 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_36 = memref.reinterpret_cast %arg23 to offset: [%232], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_37 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_36, %alloc_37 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %233 = bufferization.to_tensor %alloc_37 restrict writable {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %234 = arith.addi %218, %c255_i32 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %235 = arith.subi %234, %104 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %236 = arith.maxsi %235, %14 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %237 = arith.minsi %236, %arg24 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %238 = arith.index_cast %237 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %239 = arith.addi %222, %238 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_38 = memref.reinterpret_cast %arg23 to offset: [%239], sizes: [128, 128], strides: [%97, 1] {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_39 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_38, %alloc_39 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %240 = bufferization.to_tensor %alloc_39 restrict writable {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %241 = arith.andi %233, %240 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %242 = arith.andi %241, %100 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %243 = arith.andi %242, %226 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %244 = arith.mulf %217, %10 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %245 = arith.cmpi ne, %243, %12 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %246 = arith.select %245, %244, %9 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi1>, tensor<128x128xf32>
          %reduced = linalg.reduce ins(%246 : tensor<128x128xf32>) outs(%8 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %262 = arith.maximumf %in, %init : f32
              linalg.yield %262 : f32
            }
          %247 = arith.maximumf %arg36, %reduced {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_40 = linalg.broadcast ins(%247 : tensor<128xf32>) outs(%6 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"}
          %248 = arith.subf %246, %broadcasted_40 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %249 = math.exp %248 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %250 = arith.truncf %249 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>
          %reduced_41 = linalg.reduce ins(%249 : tensor<128x128xf32>) outs(%5 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %262 = arith.addf %in, %init {ssbuffer.block_id = 25 : i32} : f32
              linalg.yield %262 {ssbuffer.block_id = 25 : i32} : f32
            }
          %251 = arith.subf %arg36, %247 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %252 = math.exp %251 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %253 = arith.mulf %arg37, %252 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %254 = arith.addf %253, %reduced_41 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_42 = linalg.broadcast ins(%252 : tensor<128xf32>) outs(%6 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"}
          %alloc_43 = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %213 {
            linalg.fill {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_5 : bf16) outs(%alloc_43 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 14 : i32}
          %255 = arith.addi %180, %86 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_44 = memref.reinterpret_cast %arg7 to offset: [%255], sizes: [128, 128], strides: [%179, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_45 = memref.subview %reinterpret_cast_44[0, 0] [%208, %210] [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_46 = memref.subview %alloc_43[%207, %209] [%208, %210] [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_45, %subview_46 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %256 = bufferization.to_tensor %alloc_43 restrict writable {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %257 = tensor.empty() {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
          %258 = linalg.fill {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%257 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %259 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%250, %256 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%258 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %260 = arith.mulf %arg38, %broadcasted_42 {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %261 = arith.addf %259, %260 {ssbuffer.add_from_matmul, ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"} %247, %254, %261 : tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>
        } {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"}
        %115 = arith.remsi %arg34, %c16_i32_4 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %116 = arith.muli %115, %arg16 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %117 = arith.index_cast %116 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %118 = arith.index_cast %31 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %119 = arith.addi %117, %118 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %120 = arith.maxsi %89, %c0_i32_7 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %121 = arith.index_cast %120 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %122 = arith.index_cast %34 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %123 = arith.subi %c0_i32_7, %89 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %124 = arith.maxsi %123, %c0_i32_7 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %125 = arith.index_cast %124 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %126 = arith.index_cast %89 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %127 = arith.addi %119, %126 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_22 = memref.reinterpret_cast %arg4 to offset: [%127], sizes: [128], strides: [1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        %128 = arith.addi %126, %c128_2 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %129 = arith.index_cast %34 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %130 = arith.maxsi %126, %129 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %131 = arith.minsi %128, %130 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %132 = arith.subi %131, %126 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %subview_23 = memref.subview %reinterpret_cast_22[0] [%132] [1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %133 = arith.muli %115, %arg13 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %134 = arith.index_cast %133 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %135 = arith.addi %43, %134 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %136 = arith.index_cast %arg12 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %137 = arith.muli %121, %136 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %138 = arith.addi %137, %135 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_24 = memref.reinterpret_cast %arg2 to offset: [%138], sizes: [128, 128], strides: [%136, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %139 = arith.muli %115, %arg15 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %140 = arith.index_cast %139 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %141 = arith.addi %45, %140 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %142 = arith.index_cast %arg14 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %143 = arith.muli %121, %142 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %144 = arith.addi %143, %141 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_25 = memref.reinterpret_cast %arg3 to offset: [%144], sizes: [128, 128], strides: [%142, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
        %145 = arith.divsi %143, %142 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %146 = arith.subi %122, %145 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %147 = arith.maxsi %146, %c0_3 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %148 = arith.minsi %147, %c128_2 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %149 = arith.remsi %143, %142 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %150 = arith.subi %c128_2, %149 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %151 = arith.maxsi %150, %c0_3 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %152 = arith.minsi %151, %c128_2 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %153 = arith.minsi %125, %148 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %154 = arith.subi %148, %153 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %155 = arith.minsi %152, %c0_3 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %156 = arith.subi %152, %155 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %subview_26 = memref.subview %reinterpret_cast_25[0, 0] [%154, %156] [1, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %157 = arith.divsi %137, %136 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %158 = arith.subi %122, %157 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %159 = arith.maxsi %158, %c0_3 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %160 = arith.minsi %159, %c128_2 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %161 = arith.remsi %137, %136 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %162 = arith.subi %c128_2, %161 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %163 = arith.maxsi %162, %c0_3 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %164 = arith.minsi %163, %c128_2 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %165 = arith.minsi %125, %160 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %166 = arith.subi %160, %165 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %167 = arith.minsi %164, %c0_3 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %168 = arith.subi %164, %167 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %subview_27 = memref.subview %reinterpret_cast_24[0, 0] [%166, %168] [1, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %169 = math.log %114#1 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
        %170 = arith.addf %114#0, %169 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
        %extracted_slice = tensor.extract_slice %170[0] [%132] [1] {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32> to tensor<?xf32>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_23 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
        %broadcasted = linalg.broadcast ins(%114#1 : tensor<128xf32>) outs(%6 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"}
        %171 = arith.divf %114#2, %broadcasted {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %extracted_slice_28 = tensor.extract_slice %171[%153, %155] [%154, %156] [1, 1] {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<?x?xf32>
        bufferization.materialize_in_destination %extracted_slice_28 in writable %subview_26 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
        %172 = arith.truncf %171 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>
        %extracted_slice_29 = tensor.extract_slice %172[%165, %167] [%166, %168] [1, 1] {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xbf16> to tensor<?x?xbf16>
        bufferization.materialize_in_destination %extracted_slice_29 in writable %subview_27 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
      }
      scf.yield {ssbuffer.core_type = "VECTOR"} %38 : i32
    } {ssbuffer.core_type = "VECTOR"}
    return
  }
}

