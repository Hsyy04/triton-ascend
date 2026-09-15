// RUN: triton-opt --merge-high-fanout-block %s | FileCheck %s

// Producer block 20 holds 17 distinct VECTOR ops; each is consumed by an op in
// block 30 (also VECTOR). Since block 20 feeds block 30 through >16 distinct
// source ops, the pass merges producer block 20 into the biggest consumer
// block 30 (block 20's ops take block 30's id). A second consumer block 40
// consumes only 3 of those sources (<=16) and is left untouched; the merge
// fires once for this MLIR Block then stops.

// CHECK-LABEL: func.func @merge_high_fanout_block
// CHECK: arith.constant {{{.*}}ssbuffer.block_id = 30 {{.*}} 0.000000e+00
// CHECK: arith.constant {{{.*}}ssbuffer.block_id = 30 {{.*}} 1.600000e+01
// CHECK: arith.addf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 40
module {
  func.func @merge_high_fanout_block(%arg0: f32) {
    // producer block 20 (VECTOR) — 17 distinct source ops
    %p0 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 0.0 : f32
    %p1 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f32
    %p2 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 2.0 : f32
    %p3 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 3.0 : f32
    %p4 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 4.0 : f32
    %p5 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 5.0 : f32
    %p6 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 6.0 : f32
    %p7 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 7.0 : f32
    %p8 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 8.0 : f32
    %p9 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 9.0 : f32
    %p10 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 10.0 : f32
    %p11 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 11.0 : f32
    %p12 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 12.0 : f32
    %p13 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 13.0 : f32
    %p14 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 14.0 : f32
    %p15 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 15.0 : f32
    %p16 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 16.0 : f32

    // consumer block 30 (VECTOR) — each op uses one distinct source from 20
    %a0 = arith.addf %p0, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a1 = arith.addf %p1, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a2 = arith.addf %p2, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a3 = arith.addf %p3, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a4 = arith.addf %p4, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a5 = arith.addf %p5, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a6 = arith.addf %p6, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a7 = arith.addf %p7, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a8 = arith.addf %p8, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a9 = arith.addf %p9, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a10 = arith.addf %p10, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a11 = arith.addf %p11, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a12 = arith.addf %p12, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a13 = arith.addf %p13, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a14 = arith.addf %p14, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a15 = arith.addf %p15, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %a16 = arith.addf %p16, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32

    // second consumer block 40 (VECTOR) — only 3 sources, left unchanged
    %b0 = arith.addf %p0, %arg0 {ssbuffer.block_id = 40 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %b1 = arith.addf %p1, %arg0 {ssbuffer.block_id = 40 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %b2 = arith.addf %p2, %arg0 {ssbuffer.block_id = 40 : i32, ssbuffer.core_type = "VECTOR"} : f32

    %unused = arith.addf %a0, %b0 {ssbuffer.block_id = 50 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return
  }
}
