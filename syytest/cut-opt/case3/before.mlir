module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_swa_fwd_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32, %arg31: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : bf16
    %cst_0 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0xFF800000 : f32
    %c1_i32 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_1 = arith.constant {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
    %1 = linalg.fill {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_1 : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    %2 = tensor.empty() {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %3 = linalg.fill {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_1 : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %4 = linalg.fill {ssbuffer.block_id = 34 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    %cst_2 = arith.constant {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} -1.000000e+06 : f32
    %5 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_2 : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %6 = linalg.fill {ssbuffer.block_id = 39 : i32, ssbuffer.core_type = "VECTOR"} ins(%arg11 : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %c0_i8 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 0 : i8
    %c3_i32 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 3 : i32
    %c4_i32 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 4 : i32
    %c255_i32 = arith.constant {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} 255 : i32
    %7 = tensor.empty() {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
    %8 = linalg.fill {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} ins(%c0_i8 : i8) outs(%7 : tensor<128x128xi8>) -> tensor<128x128xi8>
    %9 = arith.muli %arg24, %c3_i32 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %10 = arith.subi %c0_i32, %arg24 {ssbuffer.block_id = 37 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %c128 = arith.constant {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} 128 : index
    %c0 = arith.constant {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c16_i32 = arith.constant {ssbuffer.block_id = 35 : i32, ssbuffer.core_type = "VECTOR"} 16 : i32
    %c128_3 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 128 : index
    %c1 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 1 : index
    %c0_4 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 0 : index
    %c0_i32_5 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %c16_i32_6 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 16 : i32
    %c4_i32_7 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 4 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 128 : i32
    %cst_8 = arith.constant {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %11 = tensor.empty() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %12 = linalg.fill {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_8 : f32) outs(%11 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %c1_9 = arith.constant {ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %c128_i32_10 = arith.constant {ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c127_i32 = arith.constant {ssbuffer.block_id = 36 : i32, ssbuffer.core_type = "VECTOR"} 127 : i32
    %c2_i32 = arith.constant {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} 2 : i32
    %13 = arith.muli %arg24, %arg25 {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %14 = arith.muli %arg24, %c2_i32 {ssbuffer.block_id = 38 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %15 = scf.for %arg32 = %c0_i32_5 to %arg8 step %c1_i32 iter_args(%arg33 = %c0_i32) -> (i32)  : i32 {
      %16 = arith.index_cast %arg32 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%16], sizes: [1], strides: [1] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %17 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %18 = arith.addi %16, %c1_9 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg9 to offset: [%18], sizes: [1], strides: [1] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %19 = memref.load %reinterpret_cast_11[%c0] {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %20 = arith.subi %19, %17 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %21 = arith.addi %20, %c127_i32 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %22 = arith.divsi %21, %c128_i32_10 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %23 = arith.muli %arg33, %c16_i32 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %24 = arith.addi %arg33, %22 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %25 = arith.muli %22, %c16_i32 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %26 = arith.addi %23, %arg29 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %27 = arith.remsi %26, %arg26 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %28 = arith.muli %17, %arg12 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %29 = arith.index_cast %28 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %30 = arith.muli %17, %arg14 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %31 = arith.index_cast %30 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg10 to offset: [%16], sizes: [1], strides: [1] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %32 = memref.load %reinterpret_cast_12[%c0] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %reinterpret_cast_13 = memref.reinterpret_cast %arg10 to offset: [%18], sizes: [1], strides: [1] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %33 = memref.load %reinterpret_cast_13[%c0] {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %34 = arith.subi %33, %32 {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %35 = arith.subi %34, %20 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %36 = arith.index_cast %arg32 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg9 to offset: [%36], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %37 = memref.load %reinterpret_cast_14[%c0_4] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %38 = arith.addi %36, %c1 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg9 to offset: [%38], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %39 = memref.load %reinterpret_cast_15[%c0_4] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %reinterpret_cast_16 = memref.reinterpret_cast %arg10 to offset: [%36], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %40 = memref.load %reinterpret_cast_16[%c0_4] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %reinterpret_cast_17 = memref.reinterpret_cast %arg10 to offset: [%38], sizes: [1], strides: [1] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %41 = memref.load %reinterpret_cast_17[%c0_4] {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
      %42 = arith.subi %39, %37 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %43 = arith.subi %41, %40 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %44 = arith.muli %37, %arg17 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %45 = arith.index_cast %44 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %46 = arith.muli %40, %arg19 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %47 = arith.index_cast %46 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %48 = arith.muli %40, %arg21 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32
      %49 = arith.index_cast %48 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      scf.for %arg34 = %27 to %25 step %arg26  : i32 {
        %50 = arith.divsi %arg34, %c16_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %51 = arith.muli %50, %c128_i32_10 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %52 = arith.subi %51, %20 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %53 = arith.maxsi %52, %10 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %54 = arith.minsi %53, %c0_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %55 = arith.index_cast %13 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %56 = arith.index_cast %14 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %57 = arith.addi %55, %56 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : index
        %58 = arith.index_cast %54 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %59 = arith.index_cast %arg25 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %60 = arith.muli %58, %59 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : index
        %61 = arith.addi %57, %60 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_18 = memref.reinterpret_cast %arg23 to offset: [%61], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
        memref.copy %reinterpret_cast_18, %alloc {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
        %62 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
        %63 = arith.addi %51, %c128_i32_10 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %64 = arith.minsi %63, %20 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %65 = arith.subi %64, %51 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %66 = arith.addi %51, %35 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %67 = arith.addi %66, %65 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %68 = arith.addi %67, %c127_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %69 = arith.divsi %68, %c128_i32_10 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %70 = arith.minsi %69, %c1_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %71 = arith.subi %66, %c255_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %72 = arith.maxsi %71, %c0_i32 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %73 = arith.divsi %72, %c128_i32_10 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %74 = arith.maxsi %70, %73 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %75 = arith.remsi %arg34, %c16_i32 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %76 = arith.muli %75, %arg16 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %77 = arith.index_cast %76 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %78 = arith.index_cast %17 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %79 = arith.addi %77, %78 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %80 = arith.maxsi %51, %c0_i32 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %81 = arith.index_cast %80 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %82 = arith.index_cast %20 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %83 = arith.subi %c0_i32, %51 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %84 = arith.maxsi %83, %c0_i32 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %85 = arith.index_cast %84 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %86 = arith.index_cast %51 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %87 = arith.addi %79, %86 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%87], sizes: [128], strides: [1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        %88 = arith.addi %86, %c128 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %89 = arith.index_cast %20 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %90 = arith.maxsi %86, %89 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %91 = arith.minsi %88, %90 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %92 = arith.subi %91, %86 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %subview = memref.subview %reinterpret_cast_19[0] [%92] [1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %93 = arith.muli %75, %arg13 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %94 = arith.index_cast %93 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %95 = arith.addi %29, %94 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %96 = arith.index_cast %arg12 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %97 = arith.muli %81, %96 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %98 = arith.addi %97, %95 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [128, 128], strides: [%96, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %99 = arith.muli %75, %arg15 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32
        %100 = arith.index_cast %99 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %101 = arith.addi %31, %100 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %102 = arith.index_cast %arg14 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
        %103 = arith.muli %81, %102 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %104 = arith.addi %103, %101 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [128, 128], strides: [%102, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
        %105 = arith.divsi %103, %102 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %106 = arith.subi %82, %105 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %107 = arith.maxsi %106, %c0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %108 = arith.minsi %107, %c128 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %109 = arith.remsi %103, %102 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %110 = arith.subi %c128, %109 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %111 = arith.maxsi %110, %c0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %112 = arith.minsi %111, %c128 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %113 = arith.minsi %85, %108 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %114 = arith.subi %108, %113 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %115 = arith.minsi %112, %c0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %116 = arith.subi %112, %115 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %subview_22 = memref.subview %reinterpret_cast_21[0, 0] [%114, %116] [1, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %117 = arith.divsi %97, %96 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %118 = arith.subi %82, %117 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %119 = arith.maxsi %118, %c0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %120 = arith.minsi %119, %c128 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %121 = arith.remsi %97, %96 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %122 = arith.subi %c128, %121 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %123 = arith.maxsi %122, %c0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %124 = arith.minsi %123, %c128 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %125 = arith.minsi %85, %120 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %126 = arith.subi %120, %125 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %127 = arith.minsi %124, %c0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %128 = arith.subi %124, %127 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : index
        %subview_23 = memref.subview %reinterpret_cast_20[0, 0] [%126, %128] [1, 1] {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %129 = arith.divsi %arg34, %c16_i32_6 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %130 = arith.remsi %arg34, %c16_i32_6 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %131 = arith.remsi %130, %c4_i32_7 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %132 = arith.muli %129, %c128_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %133 = arith.muli %130, %arg18 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %134 = arith.index_cast %133 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %135 = arith.addi %45, %134 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %136 = arith.maxsi %132, %c0_i32_5 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %137 = arith.index_cast %136 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %138 = arith.index_cast %arg17 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %139 = arith.muli %137, %138 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %140 = arith.addi %139, %135 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %141 = arith.index_cast %42 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %reinterpret_cast_24 = memref.reinterpret_cast %arg5 to offset: [%140], sizes: [128, 128], strides: [%138, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %142 = arith.divsi %139, %138 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %143 = arith.subi %141, %142 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %144 = arith.maxsi %143, %c0_4 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %145 = arith.minsi %144, %c128_3 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %146 = arith.remsi %139, %138 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %147 = arith.subi %c128_3, %146 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %148 = arith.maxsi %147, %c0_4 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %149 = arith.minsi %148, %c128_3 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %150 = arith.subi %c0_i32_5, %132 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %151 = arith.maxsi %150, %c0_i32_5 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %152 = arith.index_cast %151 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %153 = arith.minsi %152, %145 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %154 = arith.subi %145, %153 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %155 = arith.minsi %149, %c0_4 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %156 = arith.subi %149, %155 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %157 = arith.cmpi slt, %154, %c128_3 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %158 = arith.cmpi slt, %156, %c128_3 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %159 = arith.ori %157, %158 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i1
        %subview_25 = memref.subview %reinterpret_cast_24[0, 0] [%154, %156] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %160 = arith.muli %131, %arg20 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %161 = arith.index_cast %160 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %162 = arith.addi %47, %161 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %163 = arith.muli %131, %arg22 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32
        %164 = arith.index_cast %163 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %165 = arith.addi %49, %164 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : index
        %alloc_26 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %subview_27 = memref.subview %alloc_26[%153, %155] [%154, %156] [1, 1] {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
        scf.if %159 {
          linalg.fill {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_26 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 17 : i32}
        memref.copy %subview_25, %subview_27 {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
        %166 = bufferization.to_tensor %alloc_26 restrict writable {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %167:3 = scf.for %arg35 = %c0_i32_5 to %70 step %c1_i32 iter_args(%arg36 = %4, %arg37 = %1, %arg38 = %3) -> (tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>)  : i32 {
          %173 = arith.muli %arg35, %c128_i32_10 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %174 = arith.subi %173, %34 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %175 = arith.maxsi %174, %10 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %176 = arith.minsi %175, %c0_i32 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %177 = arith.index_cast %arg24 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %178 = arith.addi %55, %177 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %179 = arith.index_cast %176 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %180 = arith.addi %178, %179 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_30 = memref.reinterpret_cast %arg23 to offset: [%180], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_31 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_30, %alloc_31 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %181 = bufferization.to_tensor %alloc_31 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %182 = arith.subi %173, %c4_i32 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %183 = arith.maxsi %182, %10 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %184 = arith.minsi %183, %c0_i32 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %185 = arith.index_cast %184 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %186 = arith.addi %178, %185 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_32 = memref.reinterpret_cast %arg23 to offset: [%186], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_33 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_32, %alloc_33 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %187 = bufferization.to_tensor %alloc_33 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %188 = arith.addi %173, %c255_i32 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %189 = arith.subi %188, %66 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %190 = arith.maxsi %189, %10 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %191 = arith.minsi %190, %arg24 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %192 = arith.index_cast %191 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %193 = arith.addi %177, %192 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_34 = memref.reinterpret_cast %arg23 to offset: [%193], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_35 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_34, %alloc_35 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %194 = bufferization.to_tensor %alloc_35 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %195 = arith.ori %187, %194 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %196 = arith.subi %173, %66 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %197 = arith.maxsi %196, %10 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %198 = arith.minsi %197, %arg24 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %199 = arith.index_cast %9 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %200 = arith.index_cast %198 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %201 = arith.addi %199, %200 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_36 = memref.reinterpret_cast %arg23 to offset: [%201], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_37 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_36, %alloc_37 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %202 = bufferization.to_tensor %alloc_37 restrict writable {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %203 = arith.andi %195, %202 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %204 = arith.andi %203, %62 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %205 = arith.andi %204, %181 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %206 = arith.cmpi ne, %205, %8 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>



          %207 = arith.muli %arg35, %c128_i32 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %208 = arith.maxsi %207, %c0_i32_5 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %209 = arith.index_cast %208 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %210 = arith.index_cast %arg19 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %211 = arith.muli %209, %210 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %212 = arith.index_cast %43 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %213 = arith.index_cast %arg21 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %214 = arith.muli %209, %213 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %215 = arith.divsi %211, %210 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %216 = arith.subi %212, %215 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %217 = arith.maxsi %216, %c0_4 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %218 = arith.minsi %217, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %219 = arith.remsi %211, %210 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %220 = arith.subi %c128_3, %219 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %221 = arith.maxsi %220, %c0_4 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %222 = arith.minsi %221, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %223 = arith.subi %c0_i32_5, %207 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %224 = arith.maxsi %223, %c0_i32_5 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
          %225 = arith.index_cast %224 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %226 = arith.minsi %225, %218 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %227 = arith.subi %218, %226 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %228 = arith.minsi %222, %c0_4 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %229 = arith.subi %222, %228 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %230 = arith.cmpi slt, %227, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %231 = arith.cmpi slt, %229, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %232 = arith.ori %230, %231 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i1
          %233 = arith.divsi %214, %213 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %234 = arith.subi %212, %233 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %235 = arith.maxsi %234, %c0_4 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %236 = arith.minsi %235, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %237 = arith.remsi %214, %213 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %238 = arith.subi %c128_3, %237 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %239 = arith.maxsi %238, %c0_4 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %240 = arith.minsi %239, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %241 = arith.minsi %225, %236 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %242 = arith.subi %236, %241 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %243 = arith.minsi %240, %c0_4 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %244 = arith.subi %240, %243 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %245 = arith.cmpi slt, %242, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %246 = arith.cmpi slt, %244, %c128_3 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
          %247 = arith.ori %245, %246 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i1
          %alloc_38 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %232 {
            linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_38 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
          %248 = arith.addi %211, %162 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_39 = memref.reinterpret_cast %arg6 to offset: [%248], sizes: [128, 128], strides: [%210, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_40 = memref.subview %reinterpret_cast_39[0, 0] [%227, %229] [1, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_41 = memref.subview %alloc_38[%226, %228] [%227, %229] [1, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_40, %subview_41 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %249 = bufferization.to_tensor %alloc_38 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %250 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xbf16>
          %transposed = linalg.transpose ins(%249 : tensor<128x128xbf16>) outs(%250 : tensor<128x128xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"}
          %251 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%166, %transposed : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%12 : tensor<128x128xf32>) -> tensor<128x128xf32>
          
          
          
          %252 = arith.mulf %251, %6 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %253 = arith.select %206, %252, %5 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi1>, tensor<128x128xf32>
          %reduced = linalg.reduce ins(%253 : tensor<128x128xf32>) outs(%4 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %269 = arith.maximumf %in, %init : f32
              linalg.yield %269 : f32
            }
          %254 = arith.maximumf %arg36, %reduced {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_42 = linalg.broadcast ins(%254 : tensor<128xf32>) outs(%2 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"}
          %255 = arith.subf %253, %broadcasted_42 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %256 = math.exp %255 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %257 = arith.truncf %256 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>
          %reduced_43 = linalg.reduce ins(%256 : tensor<128x128xf32>) outs(%1 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %269 = arith.addf %in, %init {ssbuffer.block_id = 21 : i32} : f32
              linalg.yield %269 {ssbuffer.block_id = 21 : i32} : f32
            }
          %258 = arith.subf %arg36, %254 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>


          %259 = math.exp %258 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %260 = arith.mulf %arg37, %259 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %261 = arith.addf %260, %reduced_43 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_44 = linalg.broadcast ins(%259 : tensor<128xf32>) outs(%2 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"}
          
          %alloc_45 = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %247 {
            linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_45 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
          %262 = arith.addi %214, %165 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_46 = memref.reinterpret_cast %arg7 to offset: [%262], sizes: [128, 128], strides: [%213, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_47 = memref.subview %reinterpret_cast_46[0, 0] [%242, %244] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_48 = memref.subview %alloc_45[%241, %243] [%242, %244] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_47, %subview_48 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %263 = bufferization.to_tensor %alloc_45 restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %264 = tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
          %265 = linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_8 : f32) outs(%264 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %266 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%257, %263 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%265 : tensor<128x128xf32>) -> tensor<128x128xf32>
          
          %267 = arith.mulf %arg38, %broadcasted_44 {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %268 = arith.addf %266, %267 {ssbuffer.add_from_matmul, ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"} %254, %261, %268 : tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>
        } {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"}
        %168:3 = scf.for %arg35 = %74 to %69 step %c1_i32 iter_args(%arg36 = %167#0, %arg37 = %167#1, %arg38 = %167#2) -> (tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>)  : i32 {
          %173 = arith.muli %arg35, %c128_i32_10 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %174 = arith.subi %173, %34 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %175 = arith.maxsi %174, %10 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %176 = arith.minsi %175, %c0_i32 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %177 = arith.index_cast %arg24 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %178 = arith.addi %55, %177 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %179 = arith.index_cast %176 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %180 = arith.addi %178, %179 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_30 = memref.reinterpret_cast %arg23 to offset: [%180], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_31 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_30, %alloc_31 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %181 = bufferization.to_tensor %alloc_31 restrict writable {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %182 = arith.subi %173, %66 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %183 = arith.maxsi %182, %10 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %184 = arith.minsi %183, %arg24 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %185 = arith.index_cast %9 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %186 = arith.index_cast %184 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %187 = arith.addi %185, %186 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_32 = memref.reinterpret_cast %arg23 to offset: [%187], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_33 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_32, %alloc_33 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %188 = bufferization.to_tensor %alloc_33 restrict writable {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %189 = arith.addi %173, %c255_i32 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %190 = arith.subi %189, %66 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %191 = arith.maxsi %190, %10 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %192 = arith.minsi %191, %arg24 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32
          %193 = arith.index_cast %192 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
          %194 = arith.addi %177, %193 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : index
          %reinterpret_cast_34 = memref.reinterpret_cast %arg23 to offset: [%194], sizes: [128, 128], strides: [%59, 1] {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
          %alloc_35 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xi8>
          memref.copy %reinterpret_cast_34, %alloc_35 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
          %195 = bufferization.to_tensor %alloc_35 restrict writable {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR", was_bool_to_int8 = true} : memref<128x128xi8> to tensor<128x128xi8>
          %196 = arith.andi %188, %195 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %197 = arith.andi %196, %62 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %198 = arith.andi %197, %181 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %199 = arith.cmpi ne, %198, %8 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi8>
          %200 = arith.muli %arg35, %c128_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %201 = arith.maxsi %200, %c0_i32_5 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %202 = arith.index_cast %201 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %203 = arith.index_cast %arg19 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %204 = arith.muli %202, %203 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %205 = arith.index_cast %43 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %206 = arith.index_cast %arg21 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %207 = arith.muli %202, %206 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %208 = arith.divsi %204, %203 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %209 = arith.subi %205, %208 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %210 = arith.maxsi %209, %c0_4 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %211 = arith.minsi %210, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %212 = arith.remsi %204, %203 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %213 = arith.subi %c128_3, %212 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %214 = arith.maxsi %213, %c0_4 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %215 = arith.minsi %214, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %216 = arith.subi %c0_i32_5, %200 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %217 = arith.maxsi %216, %c0_i32_5 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32
          %218 = arith.index_cast %217 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
          %219 = arith.minsi %218, %211 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %220 = arith.subi %211, %219 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %221 = arith.minsi %215, %c0_4 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %222 = arith.subi %215, %221 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %223 = arith.cmpi slt, %220, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %224 = arith.cmpi slt, %222, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %225 = arith.ori %223, %224 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i1
          %226 = arith.divsi %207, %206 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %227 = arith.subi %205, %226 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %228 = arith.maxsi %227, %c0_4 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %229 = arith.minsi %228, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %230 = arith.remsi %207, %206 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %231 = arith.subi %c128_3, %230 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %232 = arith.maxsi %231, %c0_4 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %233 = arith.minsi %232, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %234 = arith.minsi %218, %229 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %235 = arith.subi %229, %234 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %236 = arith.minsi %233, %c0_4 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %237 = arith.subi %233, %236 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %238 = arith.cmpi slt, %235, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %239 = arith.cmpi slt, %237, %c128_3 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : index
          %240 = arith.ori %238, %239 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : i1
          %alloc_36 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %225 {
            linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_36 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 12 : i32}
          %241 = arith.addi %204, %162 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_37 = memref.reinterpret_cast %arg6 to offset: [%241], sizes: [128, 128], strides: [%203, 1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_38 = memref.subview %reinterpret_cast_37[0, 0] [%220, %222] [1, 1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_39 = memref.subview %alloc_36[%219, %221] [%220, %222] [1, 1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_38, %subview_39 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %242 = bufferization.to_tensor %alloc_36 restrict writable {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %243 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xbf16>
          %transposed = linalg.transpose ins(%242 : tensor<128x128xbf16>) outs(%243 : tensor<128x128xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"}
          %244 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%166, %transposed : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%12 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %245 = arith.mulf %244, %6 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %246 = arith.select %199, %245, %5 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi1>, tensor<128x128xf32>
          %reduced = linalg.reduce ins(%246 : tensor<128x128xf32>) outs(%4 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %262 = arith.maximumf %in, %init : f32
              linalg.yield %262 : f32
            }
          %247 = arith.maximumf %arg36, %reduced {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_40 = linalg.broadcast ins(%247 : tensor<128xf32>) outs(%2 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"}
          %248 = arith.subf %246, %broadcasted_40 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %249 = math.exp %248 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %250 = arith.truncf %249 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>
          %reduced_41 = linalg.reduce ins(%249 : tensor<128x128xf32>) outs(%1 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
              %262 = arith.addf %in, %init {ssbuffer.block_id = 25 : i32} : f32
              linalg.yield %262 {ssbuffer.block_id = 25 : i32} : f32
            }
          %251 = arith.subf %arg36, %247 {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %252 = math.exp %251 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %253 = arith.mulf %arg37, %252 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %254 = arith.addf %253, %reduced_41 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
          %broadcasted_42 = linalg.broadcast ins(%252 : tensor<128xf32>) outs(%2 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"}
          %alloc_43 = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
          scf.if %240 {
            linalg.fill {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%alloc_43 : memref<128x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 14 : i32}
          %255 = arith.addi %207, %165 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : index
          %reinterpret_cast_44 = memref.reinterpret_cast %arg7 to offset: [%255], sizes: [128, 128], strides: [%206, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %subview_45 = memref.subview %reinterpret_cast_44[0, 0] [%235, %237] [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_46 = memref.subview %alloc_43[%234, %236] [%235, %237] [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_45, %subview_46 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
          %256 = bufferization.to_tensor %alloc_43 restrict writable {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
          %257 = tensor.empty() {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
          %258 = linalg.fill {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_8 : f32) outs(%257 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %259 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%250, %256 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%258 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %260 = arith.mulf %arg38, %broadcasted_42 {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          %261 = arith.addf %259, %260 {ssbuffer.add_from_matmul, ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
          scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"} %247, %254, %261 : tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>
        } {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"}
        %169 = math.log %168#1 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
        %170 = arith.addf %168#0, %169 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
        %extracted_slice = tensor.extract_slice %170[0] [%92] [1] {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32> to tensor<?xf32>
        bufferization.materialize_in_destination %extracted_slice in writable %subview {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
        %broadcasted = linalg.broadcast ins(%168#1 : tensor<128xf32>) outs(%2 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"}
        %171 = arith.divf %168#2, %broadcasted {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %extracted_slice_28 = tensor.extract_slice %171[%113, %115] [%114, %116] [1, 1] {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<?x?xf32>
        bufferization.materialize_in_destination %extracted_slice_28 in writable %subview_22 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
        %172 = arith.truncf %171 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32> to tensor<128x128xbf16>
        %extracted_slice_29 = tensor.extract_slice %172[%125, %127] [%126, %128] [1, 1] {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xbf16> to tensor<?x?xbf16>
        bufferization.materialize_in_destination %extracted_slice_29 in writable %subview_23 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
      }
      scf.yield {ssbuffer.core_type = "VECTOR"} %24 : i32
    } {ssbuffer.core_type = "VECTOR"}
    return
  }
}

