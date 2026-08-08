module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.load_store_buf_count = 1 : i32} {
  func.func @flex_attention_backward_dq_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xi8> {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg15: memref<?xi32> {tt.divisibility = 16 : i32}, %arg16: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg21: i32, %arg22: i32, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32, %arg31: i32, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: i32, %arg41: i32, %arg42: i32, %arg43: i32 {tt.divisibility = 16 : i32}, %arg44: i32, %arg45: i32, %arg46: i32, %arg47: i32, %arg48: i32, %arg49: i32, %arg50: i32, %arg51: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c128 = arith.constant 128 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %cst_0 = arith.constant 0.0883883461 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = tensor.empty() : tensor<128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %6 = tensor.empty() : tensor<128x128xi8>
    %7 = linalg.fill ins(%c0_i8 : i8) outs(%6 : tensor<128x128xi8>) -> tensor<128x128xi8>
    %8 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %9 = arith.extsi %arg21 : i32 to i64
    %10 = arith.extsi %arg22 : i32 to i64
    %11 = arith.extsi %arg24 : i32 to i64
    %12 = arith.extsi %arg25 : i32 to i64
    %13 = arith.extsi %arg27 : i32 to i64
    %14 = arith.extsi %arg28 : i32 to i64
    %15 = arith.extsi %arg30 : i32 to i64
    %16 = arith.extsi %arg31 : i32 to i64
    %17 = arith.extsi %arg33 : i32 to i64
    %18 = arith.extsi %arg34 : i32 to i64
    %19 = arith.extsi %arg35 : i32 to i64
    %20 = arith.extsi %arg36 : i32 to i64
    %21 = arith.extsi %arg37 : i32 to i64
    %22 = arith.extsi %arg38 : i32 to i64
    scf.for %arg52 = %arg49 to %arg41 step %arg46  : i32 {
      %23 = arith.remsi %arg52, %arg42 : i32
      %24 = arith.divsi %arg52, %arg42 : i32
      %25 = arith.divsi %24, %arg43 : i32
      %26 = arith.remsi %24, %arg43 : i32
      %27 = arith.divsi %26, %c2_i32 : i32
      %28 = arith.extsi %25 : i32 to i64
      %29 = arith.extsi %26 : i32 to i64
      %30 = arith.extsi %27 : i32 to i64
      %31 = arith.muli %28, %9 : i64
      %32 = arith.muli %29, %10 : i64
      %33 = arith.addi %31, %32 : i64
      %34 = arith.muli %28, %11 : i64
      %35 = arith.muli %30, %12 : i64
      %36 = arith.addi %34, %35 : i64
      %37 = arith.muli %28, %13 : i64
      %38 = arith.muli %30, %14 : i64
      %39 = arith.addi %37, %38 : i64
      %40 = arith.muli %28, %15 : i64
      %41 = arith.muli %29, %16 : i64
      %42 = arith.addi %40, %41 : i64
      %43 = arith.muli %28, %17 : i64
      %44 = arith.muli %29, %18 : i64
      %45 = arith.addi %43, %44 : i64
      %46 = arith.muli %28, %19 : i64
      %47 = arith.muli %29, %20 : i64
      %48 = arith.addi %46, %47 : i64
      %49 = arith.muli %28, %21 : i64
      %50 = arith.muli %29, %22 : i64
      %51 = arith.addi %49, %50 : i64
      %52 = arith.index_cast %33 : i64 to index
      %53 = arith.index_cast %36 : i64 to index
      %54 = arith.index_cast %39 : i64 to index
      %55 = arith.index_cast %42 : i64 to index
      %56 = arith.index_cast %45 : i64 to index
      %57 = arith.index_cast %48 : i64 to index
      %58 = arith.index_cast %51 : i64 to index
      %59 = arith.muli %23, %c128_i32 : i32
      %60 = arith.index_cast %59 : i32 to index
      %61 = arith.index_cast %arg23 : i32 to index
      %62 = arith.muli %60, %61 : index
      %63 = arith.addi %52, %62 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%63], sizes: [128, 128], strides: [%61, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xbf16>
      %64 = arith.addi %60, %c128 : index
      %65 = arith.index_cast %arg44 : i32 to index
      %66 = arith.maxsi %60, %65 : index
      %67 = arith.minsi %64, %66 : index
      %68 = arith.subi %67, %60 : index
      %69 = arith.cmpi slt, %68, %c128 : index
      scf.if %69 {
        linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x128xbf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0, 0] [%68, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [%68, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %subview, %subview_3 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %70 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
      annotation.mark %70 {cv_pipeline_lazy_load = true} : tensor<128x128xbf16>
      %71 = arith.index_cast %arg32 : i32 to index
      %72 = arith.muli %60, %71 : index
      %73 = arith.addi %55, %72 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%73], sizes: [128, 128], strides: [%71, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
      %alloc_5 = memref.alloc() : memref<128x128xbf16>
      scf.if %69 {
        linalg.fill ins(%cst : bf16) outs(%alloc_5 : memref<128x128xbf16>)
      } {hivm.unlikely_condition}
      %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%68, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%68, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %subview_6, %subview_7 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %74 = bufferization.to_tensor %alloc_5 restrict writable : memref<128x128xbf16>
      annotation.mark %74 {cv_pipeline_lazy_load = true} : tensor<128x128xbf16>
      %75 = arith.addi %56, %60 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%75], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      %alloc_9 = memref.alloc() : memref<128xf32>
      scf.if %69 {
        linalg.fill ins(%cst_1 : f32) outs(%alloc_9 : memref<128xf32>)
      } {hivm.unlikely_condition}
      %subview_10 = memref.subview %reinterpret_cast_8[0] [%68] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_11 = memref.subview %alloc_9[0] [%68] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_10, %subview_11 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %76 = bufferization.to_tensor %alloc_9 restrict writable : memref<128xf32>
      %77 = arith.addi %57, %60 : index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg7 to offset: [%77], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<128xf32>
      scf.if %69 {
        linalg.fill ins(%cst_2 : f32) outs(%alloc_13 : memref<128xf32>)
      } {hivm.unlikely_condition}
      %subview_14 = memref.subview %reinterpret_cast_12[0] [%68] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_15 = memref.subview %alloc_13[0] [%68] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_14, %subview_15 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %78 = bufferization.to_tensor %alloc_13 restrict writable : memref<128xf32>
      %79 = arith.cmpf oeq, %76, %3 : tensor<128xf32>
      %80 = arith.select %79, %4, %76 : tensor<128xi1>, tensor<128xf32>
      %81 = arith.muli %23, %arg40 : i32
      %82 = arith.index_cast %81 : i32 to index
      %83 = arith.index_cast %23 : i32 to index
      %reinterpret_cast_16 = memref.reinterpret_cast %arg8 to offset: [%83], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %84 = memref.load %reinterpret_cast_16[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %85 = arith.muli %23, %arg19 : i32
      %86 = arith.index_cast %85 : i32 to index
      %broadcasted = linalg.broadcast ins(%80 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1] 
      %broadcasted_17 = linalg.broadcast ins(%78 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1] 
      %87 = scf.for %arg53 = %c0_i32 to %84 step %c1_i32 iter_args(%arg54 = %1) -> (tensor<128x128xf32>)  : i32 {
        %96 = arith.index_cast %arg53 : i32 to index
        %97 = arith.addi %82, %96 : index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg9 to offset: [%97], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %98 = memref.load %reinterpret_cast_21[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %99 = arith.muli %98, %c128_i32 : i32
        %100 = arith.index_cast %99 : i32 to index
        %101 = arith.index_cast %arg26 : i32 to index
        %102 = arith.muli %100, %101 : index
        %103 = arith.addi %53, %102 : index
        %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [128, 128], strides: [%101, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %alloc_23 = memref.alloc() : memref<128x128xbf16>
        %104 = arith.addi %100, %c128 : index
        %105 = arith.index_cast %arg45 : i32 to index
        %106 = arith.maxsi %100, %105 : index
        %107 = arith.minsi %104, %106 : index
        %108 = arith.subi %107, %100 : index
        %109 = arith.cmpi slt, %108, %c128 : index
        scf.if %109 {
          linalg.fill ins(%cst : bf16) outs(%alloc_23 : memref<128x128xbf16>)
        } {hivm.unlikely_condition}
        %subview_24 = memref.subview %reinterpret_cast_22[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_25 = memref.subview %alloc_23[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_24, %subview_25 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %110 = bufferization.to_tensor %alloc_23 restrict writable : memref<128x128xbf16>
        annotation.mark %110 {cv_pipeline_lazy_load = true} : tensor<128x128xbf16>
        %111 = arith.index_cast %arg29 : i32 to index
        %112 = arith.muli %100, %111 : index
        %113 = arith.addi %54, %112 : index
        %reinterpret_cast_26 = memref.reinterpret_cast %arg4 to offset: [%113], sizes: [128, 128], strides: [%111, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %alloc_27 = memref.alloc() : memref<128x128xbf16>
        scf.if %109 {
          linalg.fill ins(%cst : bf16) outs(%alloc_27 : memref<128x128xbf16>)
        } {hivm.unlikely_condition}
        %subview_28 = memref.subview %reinterpret_cast_26[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_29 = memref.subview %alloc_27[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_28, %subview_29 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %114 = bufferization.to_tensor %alloc_27 restrict writable : memref<128x128xbf16>
        annotation.mark %114 {cv_pipeline_lazy_load = true} : tensor<128x128xbf16>
        %115 = tensor.empty() : tensor<128x128xbf16>
        %transposed = linalg.transpose ins(%110 : tensor<128x128xbf16>) outs(%115 : tensor<128x128xbf16>) permutation = [1, 0] 
        %116 = linalg.matmul {input_precision = "ieee"} ins(%70, %transposed : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %117 = arith.mulf %116, %5 : tensor<128x128xf32>
        %118 = arith.index_cast %98 : i32 to index
        %119 = arith.addi %86, %118 : index
        %reinterpret_cast_30 = memref.reinterpret_cast %arg16 to offset: [%119], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %120 = memref.load %reinterpret_cast_30[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %121 = arith.maxsi %120, %c0_i32 : i32
        %122 = arith.muli %121, %arg17 : i32
        %123 = arith.index_cast %122 : i32 to index
        %124 = arith.index_cast %arg18 : i32 to index
        %reinterpret_cast_31 = memref.reinterpret_cast %arg14 to offset: [%123], sizes: [128, 128], strides: [%124, 1] : memref<?xi8> to memref<128x128xi8, strided<[?, 1], offset: ?>>
        %alloc_32 = memref.alloc() : memref<128x128xi8>
        memref.copy %reinterpret_cast_31, %alloc_32 {was_bool_to_int8 = true} : memref<128x128xi8, strided<[?, 1], offset: ?>> to memref<128x128xi8>
        %125 = bufferization.to_tensor %alloc_32 restrict writable {was_bool_to_int8 = true} : memref<128x128xi8>
        %126 = arith.cmpi sge, %120, %c0_i32 : i32
        %127 = arith.extui %126 : i1 to i8
        %128 = linalg.fill ins(%127 : i8) outs(%6 : tensor<128x128xi8>) -> tensor<128x128xi8>
        %129 = arith.andi %125, %128 : tensor<128x128xi8>
        %130 = arith.cmpi ne, %129, %7 : tensor<128x128xi8>
        %131 = arith.select %130, %117, %8 : tensor<128x128xi1>, tensor<128x128xf32>
        %132 = arith.subf %131, %broadcasted : tensor<128x128xf32>
        %133 = math.exp %132 : tensor<128x128xf32>
        %transposed_33 = linalg.transpose ins(%114 : tensor<128x128xbf16>) outs(%115 : tensor<128x128xbf16>) permutation = [1, 0] 
        %134 = linalg.matmul {input_precision = "ieee"} ins(%74, %transposed_33 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %135 = arith.subf %134, %broadcasted_17 : tensor<128x128xf32>
        %136 = arith.mulf %133, %135 : tensor<128x128xf32>
        %137 = arith.mulf %136, %5 : tensor<128x128xf32>
        %138 = arith.truncf %137 : tensor<128x128xf32> to tensor<128x128xbf16>
        %139 = linalg.matmul {input_precision = "ieee"} ins(%138, %110 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%arg54 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scf.yield %139 : tensor<128x128xf32>
      }
      %reinterpret_cast_18 = memref.reinterpret_cast %arg10 to offset: [%83], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %88 = memref.load %reinterpret_cast_18[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %89 = scf.for %arg53 = %c0_i32 to %88 step %c1_i32 iter_args(%arg54 = %87) -> (tensor<128x128xf32>)  : i32 {
        %96 = arith.index_cast %arg53 : i32 to index
        %97 = arith.addi %82, %96 : index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg11 to offset: [%97], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %98 = memref.load %reinterpret_cast_21[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %99 = arith.muli %98, %c128_i32 : i32
        %100 = arith.index_cast %99 : i32 to index
        %101 = arith.index_cast %arg26 : i32 to index
        %102 = arith.muli %100, %101 : index
        %103 = arith.addi %53, %102 : index
        %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [128, 128], strides: [%101, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %alloc_23 = memref.alloc() : memref<128x128xbf16>
        %104 = arith.addi %100, %c128 : index
        %105 = arith.index_cast %arg45 : i32 to index
        %106 = arith.maxsi %100, %105 : index
        %107 = arith.minsi %104, %106 : index
        %108 = arith.subi %107, %100 : index
        %109 = arith.cmpi slt, %108, %c128 : index
        scf.if %109 {
          linalg.fill ins(%cst : bf16) outs(%alloc_23 : memref<128x128xbf16>)
        } {hivm.unlikely_condition}
        %subview_24 = memref.subview %reinterpret_cast_22[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_25 = memref.subview %alloc_23[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_24, %subview_25 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %110 = bufferization.to_tensor %alloc_23 restrict writable : memref<128x128xbf16>
        annotation.mark %110 {cv_pipeline_lazy_load = true} : tensor<128x128xbf16>
        %111 = arith.index_cast %arg29 : i32 to index
        %112 = arith.muli %100, %111 : index
        %113 = arith.addi %54, %112 : index
        %reinterpret_cast_26 = memref.reinterpret_cast %arg4 to offset: [%113], sizes: [128, 128], strides: [%111, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %alloc_27 = memref.alloc() : memref<128x128xbf16>
        scf.if %109 {
          linalg.fill ins(%cst : bf16) outs(%alloc_27 : memref<128x128xbf16>)
        } {hivm.unlikely_condition}
        %subview_28 = memref.subview %reinterpret_cast_26[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_29 = memref.subview %alloc_27[0, 0] [%108, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_28, %subview_29 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %114 = bufferization.to_tensor %alloc_27 restrict writable : memref<128x128xbf16>
        annotation.mark %114 {cv_pipeline_lazy_load = true} : tensor<128x128xbf16>
        %115 = tensor.empty() : tensor<128x128xbf16>
        %transposed = linalg.transpose ins(%110 : tensor<128x128xbf16>) outs(%115 : tensor<128x128xbf16>) permutation = [1, 0] 
        %116 = linalg.matmul {input_precision = "ieee"} ins(%70, %transposed : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %117 = arith.mulf %116, %5 : tensor<128x128xf32>
        %118 = arith.subf %117, %broadcasted : tensor<128x128xf32>
        %119 = math.exp %118 : tensor<128x128xf32>
        %transposed_30 = linalg.transpose ins(%114 : tensor<128x128xbf16>) outs(%115 : tensor<128x128xbf16>) permutation = [1, 0] 
        %120 = linalg.matmul {input_precision = "ieee"} ins(%74, %transposed_30 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %121 = arith.subf %120, %broadcasted_17 : tensor<128x128xf32>
        %122 = arith.mulf %119, %121 : tensor<128x128xf32>
        %123 = arith.mulf %122, %5 : tensor<128x128xf32>
        %124 = arith.truncf %123 : tensor<128x128xf32> to tensor<128x128xbf16>
        %125 = linalg.matmul {input_precision = "ieee"} ins(%124, %110 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%arg54 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scf.yield %125 : tensor<128x128xf32>
      }
      %90 = arith.index_cast %arg39 : i32 to index
      %91 = arith.muli %60, %90 : index
      %92 = arith.addi %58, %91 : index
      %reinterpret_cast_19 = memref.reinterpret_cast %arg20 to offset: [%92], sizes: [128, 128], strides: [%90, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
      %93 = arith.truncf %89 : tensor<128x128xf32> to tensor<128x128xbf16>
      %94 = arith.minsi %68, %c128 : index
      %95 = arith.maxsi %94, %c0 : index
      %extracted_slice = tensor.extract_slice %93[0, 0] [%95, 128] [1, 1] : tensor<128x128xbf16> to tensor<?x128xbf16>
      %subview_20 = memref.subview %reinterpret_cast_19[0, 0] [%95, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_20 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}