// RUN: xla-translate --stablehlo-to-hlo-text -split-input-file %s | FileCheck %s
// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false -split-input-file %s | FileCheck %s --check-prefix CHECK-DIRECT

// Tests for all stablehlo ops to validate stablehlo -> hlo conversion.


// CHECK-LABEL: HloModule

// CHECK: %[[ARG0:.*]] = f32[4] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[4] parameter(1)
// CHECK: ROOT %add.3 = f32[4] add(%[[ARG0]], %[[ARG1]])
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>  func.return %0 : tensor<4xf32>
}
// CHECK-DIRECT: stablehlo.add

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[3,4]{1,0})->s32[1,2]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[3,4] parameter(0)
// CHECK-NEXT:  ROOT %[[slice_2:[^ ]+]] = s32[1,2] slice(%[[Arg_0_1]]), slice={[1:2:1], [0:4:2]},
func.func @main(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = stablehlo.slice %arg0 [1:2, 0:4:2] : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}
// CHECK-DIRECT: stablehlo.slice

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[3,4]{1,0}, s64[], s64[])->s32[1,4]{1,0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[3,4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s64[] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s64[] parameter(2)
// CHECK-NEXT:  ROOT %[[dynamic_slice_4:[^ ]+]] = s32[1,4] dynamic-slice(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), dynamic_slice_sizes={1,4},
func.func @main(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [1, 4] : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}
// CHECK-DIRECT: stablehlo.dynamic_slice

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[4]{0})->s32[1,2,3,4]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[4] parameter(0)
// CHECK-NEXT:  ROOT %[[broadcast_2:[^ ]+]] = s32[1,2,3,4] broadcast(%[[Arg_0_1]]), dimensions={3}
func.func @main(%arg0: tensor<4xi32>) -> tensor<1x2x3x4xi32> {
  %0 = stablehlo.broadcast %arg0, sizes = [1, 2, 3] : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %0 : tensor<1x2x3x4xi32>
}
// CHECK-DIRECT: stablehlo.broadcast

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[1]{0})->f32[1,10]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[1] parameter(0)
// CHECK-NEXT:  ROOT %[[broadcast_2:[^ ]+]] = f32[1,10] broadcast(%[[Arg_0_1]]), dimensions={0}
func.func @main(%arg0: tensor<1xf32>) -> tensor<1x10xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
}
// CHECK-DIRECT: stablehlo.broadcast_in_dim

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[100,26,26,32]{3,2,1,0}, f32[3,3,1,32]{3,2,1,0})->f32[100,28,28,1]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[100,26,26,32] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[3,3,1,32] parameter(1)
// CHECK-NEXT:  ROOT %[[convolution_3:[^ ]+]] = f32[100,28,28,1] convolution(%[[Arg_0_1]], %[[Arg_1_2]]), window={size=3x3 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f, metadata=

module {
  func.func @main(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32>
  return %0 : tensor<100x28x28x1xf32>
  }
}
// CHECK-DIRECT: stablehlo.convolution

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s8[100,26,26,32]{3,2,1,0}, s8[3,3,1,32]{3,2,1,0})->s32[100,28,28,1]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s8[100,26,26,32] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s8[3,3,1,32] parameter(1)
// CHECK-NEXT:  ROOT %[[convolution_3:[^ ]+]] = s32[100,28,28,1] convolution(%[[Arg_0_1]], %[[Arg_1_2]]), window={size=3x3 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f, metadata=

module {
  func.func @main(%arg0: tensor<100x26x26x32xi8>, %arg1: tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  return %0 : tensor<100x28x28x1xi32>
  }
}
// CHECK-DIRECT: stablehlo.convolution

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s8[100,26,26,32]{3,2,1,0}, s8[3,3,1,32]{3,2,1,0})->s32[100,28,28,1]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s8[100,26,26,32] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s8[3,3,1,32] parameter(1)
// CHECK-NEXT:  ROOT %[[convolution_3:[^ ]+]] = s32[100,28,28,1] convolution(%[[Arg_0_1]], %[[Arg_1_2]]), window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1}, dim_labels=b01f_01oi->b01f, metadata=

module {
  func.func @main(%arg0: tensor<100x26x26x32xi8>, %arg1: tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [true, true]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  return %0 : tensor<100x28x28x1xi32>
  }
}
// CHECK-DIRECT: stablehlo.convolution

// -----
// Binary elementwise ops

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,2]{1,0}, f32[2,2]{1,0})->f32[2,2]{1,0}}

// CHECK:       ENTRY %[[$main_11:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[2,2] parameter(1)
// CHECK-NEXT:  %[[add_3:[^ ]+]] = f32[2,2] add(%[[Arg_0_1]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[atan2_4:[^ ]+]] = f32[2,2] atan2(%[[add_3]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[divide_5:[^ ]+]] = f32[2,2] divide(%[[atan2_4]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[maximum_6:[^ ]+]] = f32[2,2] maximum(%[[divide_5]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[minimum_7:[^ ]+]] = f32[2,2] minimum(%[[maximum_6]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[multiply_8:[^ ]+]] = f32[2,2] multiply(%[[minimum_7]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[power_9:[^ ]+]] = f32[2,2] power(%[[multiply_8]], %[[Arg_1_2]]),
// CHECK-NEXT:  ROOT %[[subtract_10:[^ ]+]] = f32[2,2] subtract(%[[power_9]], %[[Arg_1_2]]),

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  %1 = stablehlo.atan2 %0, %arg1 : tensor<2x2xf32>
  %2 = stablehlo.divide %1, %arg1 : tensor<2x2xf32>
  %3 = stablehlo.maximum %2, %arg1 : tensor<2x2xf32>
  %4 = stablehlo.minimum %3, %arg1 : tensor<2x2xf32>
  %5 = stablehlo.multiply %4, %arg1 : tensor<2x2xf32>
  %6 = stablehlo.power %5, %arg1 : tensor<2x2xf32>
  %7 = stablehlo.subtract %6, %arg1 : tensor<2x2xf32>
  return %7 : tensor<2x2xf32>
}
// CHECK-DIRECT: stablehlo.add
// CHECK-DIRECT: stablehlo.atan2
// CHECK-DIRECT: stablehlo.divide
// CHECK-DIRECT: stablehlo.maximum
// CHECK-DIRECT: stablehlo.minimum
// CHECK-DIRECT: stablehlo.multiply
// CHECK-DIRECT: stablehlo.power
// CHECK-DIRECT: stablehlo.subtract

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[128,32]{1,0})->f32[128,128]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[128,32] parameter(0)
// CHECK-NEXT:  ROOT %[[all_gather_2:[^ ]+]] = f32[128,128] all-gather(%[[Arg_0_1]]), channel_id=1,
// CHECK-SAME{{LITERAL}}: replica_groups={{0,2,4,6},{1,3,5,7}},
// CHECK-SAME: dimensions={1},
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
%0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>}> {shard_count = 4 : i64} : (tensor<128x32xf32>) -> tensor<128x128xf32>
return %0 : tensor<128x128xf32>
}
// CHECK-DIRECT: stablehlo.all_gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[128,32]{1,0})->f32[128,128]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[128,32] parameter(0)
// CHECK-NEXT:  ROOT %[[all_gather_2:[^ ]+]] = f32[128,128] all-gather(%[[Arg_0_1]]), channel_id=1,
// CHECK-SAME{{LITERAL}}: replica_groups={{0,2,4,6},{1,3,5,7}},
// CHECK-SAME: dimensions={1}, use_global_device_ids=true,
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> {shard_count = 4 : i64} : (tensor<128x32xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
// CHECK-DIRECT: stablehlo.all_gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8,2]{1,0}, f32[8,4]{1,0})->(f32[8,8]{1,0}, f32[8,16]{1,0})}

// CHECK:       ENTRY %[[$main_10:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[8,4] parameter(1)
// CHECK-NEXT:  %[[tuple_3:[^ ]+]] = (f32[8,2], f32[8,4]) tuple(%[[Arg_0_1]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[get_tuple_element_4:[^ ]+]] = f32[8,2] get-tuple-element(%[[tuple_3]]), index=0,
// CHECK-NEXT:  %[[get_tuple_element_5:[^ ]+]] = f32[8,4] get-tuple-element(%[[tuple_3]]), index=1,
// CHECK-NEXT:  %[[all_gather_6:[^ ]+]] = (f32[8,8], f32[8,16]) all-gather(%[[get_tuple_element_4]], %[[get_tuple_element_5]]), channel_id=1,
// CHECK-SAME{{LITERAL}}: replica_groups={{0,2,4,6},{1,3,5,7}},
// CHECK-SAME:  dimensions={1}, use_global_device_ids=true,
// CHECK-NEXT:  %[[get_tuple_element_7:[^ ]+]] = f32[8,8] get-tuple-element(%[[all_gather_6]]), index=0,
// CHECK-NEXT:  %[[get_tuple_element_8:[^ ]+]] = f32[8,16] get-tuple-element(%[[all_gather_6]]), index=1,
// CHECK-NEXT:  ROOT %[[tuple_9:[^ ]+]] = (f32[8,8], f32[8,16]) tuple(%[[get_tuple_element_7]], %[[get_tuple_element_8]]),
func.func @main(%arg0: tensor<8x2xf32>, %arg1: tensor<8x4xf32>) -> tuple<tensor<8x8xf32>, tensor<8x16xf32>> {
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<8x2xf32>, tensor<8x4xf32>) -> (tensor<8x8xf32>, tensor<8x16xf32>)
  %1 = stablehlo.tuple %0#0, %0#1 {xla_shape = "(f32[8,8]{0,1}, f32[8,16]{0,1})"} : tuple<tensor<8x8xf32>, tensor<8x16xf32>>
  return %1 : tuple<tensor<8x8xf32>, tensor<8x16xf32>>
}
// CHECK-DIRECT: stablehlo.all_gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[4]{0}, pred[4]{0})->pred[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = pred[4] parameter(1)
// CHECK-NEXT:  ROOT %[[and_3:[^ ]+]] = pred[4] and(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-DIRECT: stablehlo.and

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16,16,16,16]{3,2,1,0}, f32[16]{0}, f32[16]{0}, f32[16]{0}, f32[16]{0})->f32[16,16,16,16]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16,16,16,16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[16] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[16] parameter(2)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = f32[16] parameter(3)
// CHECK-NEXT:  %[[Arg_4_5:[^ ]+]] = f32[16] parameter(4)
// CHECK-NEXT:  ROOT %[[batch_norm_inference_6:[^ ]+]] = f32[16,16,16,16] batch-norm-inference(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]], %[[Arg_3_4]], %[[Arg_4_5]]), epsilon=0.001, feature_index=0,
func.func @main(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
  epsilon = 0.001 : f32,
  feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}
// CHECK-DIRECT: stablehlo.batch_norm_inference

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[3,3]{1,0})->f32[3,3]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[3,3] parameter(0)
// CHECK-NEXT:  ROOT %[[cholesky_2:[^ ]+]] = f32[3,3] cholesky(%[[Arg_0_1]]), lower=true,
func.func @main(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = stablehlo.cholesky %arg0 {lower = true} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
// CHECK-DIRECT: stablehlo.cholesky

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16,8]{1,0})->f32[16,8]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16,8] parameter(0)
// CHECK-NEXT:  ROOT %[[collective_permute_2:[^ ]+]] = f32[16,8] collective-permute(%[[Arg_0_1]]),
// CHECK-SAME{{LITERAL}}: source_target_pairs={{0,1},{1,2},{2,3}},
func.func @main(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = "stablehlo.collective_permute"(%arg0) {
  source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}
// CHECK-DIRECT: stablehlo.collective_permute

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8]{0}, f32[8]{0})->f32[16]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[8] parameter(1)
// CHECK-NEXT:  ROOT %[[concatenate_3:[^ ]+]] = f32[16] concatenate(%[[Arg_0_1]], %[[Arg_1_2]]), dimensions={0},

func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<16xf32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {
  dimension = 0 : i64
  } : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.concatenate

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[])->f32[]}

// CHECK:       %[[$sum_2:[^ ]+]]
// CHECK-NEXT:  %[[x_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[y_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[add_5:[^ ]+]] = f32[] add(%[[x_3]], %[[y_4]])

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  ROOT %[[all_reduce_6:[^ ]+]] = f32[] all-reduce(%[[Arg_0_1]]),
// CHECK-SAME{{LITERAL}}: replica_groups={{0},{1}},
// CHECK-SAME: to_apply=%[[$sum_2]],
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.cross-replica-sum"(%arg0) {
  replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-DIRECT: stablehlo.cross-replica-sum

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16]{0}, f32[4]{0}, s64[])->f32[16]{0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[4] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s64[] parameter(2)
// CHECK-NEXT:  ROOT %[[dynamic_update_slice_4:[^ ]+]] = f32[16] dynamic-update-slice(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]),
func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<4xf32>, %arg2: tensor<i64>) -> tensor<16xf32> {
  %0 = "stablehlo.dynamic_update_slice"(%arg0, %arg1, %arg2) : (tensor<16xf32>, tensor<4xf32>, tensor<i64>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.dynamic_update_slice

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8,16]{1,0}, f32[16,8]{1,0})->f32[8,8]{1,0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8,16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[16,8] parameter(1)
// CHECK-NEXT:  %[[dot_3:[^ ]+]] = f32[8,8] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={grad_x="false",grad_y="false"},
// CHECK-NEXT:  ROOT %[[transpose_4:[^ ]+]] = f32[8,8] transpose(%[[dot_3]]), dimensions={0,1},
func.func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {
  einsum_config = "ab,bc->ac"
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-DIRECT: stablehlo.einsum

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(c64[16]{0})->c64[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = c64[16] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = c64[16] fft(%[[Arg_0_1]]), fft_type=FFT, fft_length={16},
func.func @main(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type FFT>,
  fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(c64[16]{0})->c64[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = c64[16] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = c64[16] fft(%[[Arg_0_1]]), fft_type=IFFT, fft_length={16},
func.func @main(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type IFFT>,
  fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16]{0})->c64[9]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = c64[9] fft(%[[Arg_0_1]]), fft_type=RFFT, fft_length={16},
func.func @main(%arg0: tensor<16xf32>) -> tensor<9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type RFFT>,
  fft_length = array<i64: 16>
  } : (tensor<16xf32>) -> tensor<9xcomplex<f32>>
  func.return %0 : tensor<9xcomplex<f32>>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(c64[9]{0})->f32[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = c64[9] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = f32[16] fft(%[[Arg_0_1]]), fft_type=IRFFT, fft_length={16},
func.func @main(%arg0: tensor<9xcomplex<f32>>) -> tensor<16xf32> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type IRFFT>,
  fft_length = array<i64: 16>
  } : (tensor<9xcomplex<f32>>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[?]{0})->s32[]}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[?] parameter(0)
// CHECK-NEXT:  ROOT %[[get_dimension_size_2:[^ ]+]] = s32[] get-dimension-size(%[[Arg_0_1]]), dimensions={0},

func.func @main(%arg0: tensor<?xf32>) -> tensor<i32> {
  %0 = "stablehlo.get_dimension_size"(%arg0) {
  dimension = 0 : i64
  } : (tensor<?xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-DIRECT: stablehlo.get_dimension_size

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[4]{0}, pred[4]{0})->pred[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = pred[4] parameter(1)
// CHECK-NEXT:  ROOT %[[or_3:[^ ]+]] = pred[4] or(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-DIRECT: stablehlo.or

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[4]{0}, s32[4]{0})->s32[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[4] parameter(1)
// CHECK-NEXT:  ROOT %[[or_3:[^ ]+]] = s32[4] or(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}
// CHECK-DIRECT: stablehlo.or

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={()->u32[]}

// CHECK:       ENTRY %[[$main_2:[^ ]+]]
// CHECK-NEXT:  ROOT %[[replica_id_1:[^ ]+]] = u32[] replica-id(),
func.func @main() -> tensor<ui32> {
  %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-DIRECT: stablehlo.replica_id

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16]{0})->f32[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16] parameter(0)
// CHECK-NEXT:  ROOT %[[reverse_2:[^ ]+]] = f32[16] reverse(%[[Arg_0_1]]), dimensions={0},
func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = "stablehlo.reverse"(%arg0) {
  dimensions = array<i64: 0>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.reverse

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[], s32[2,3]{1,0}, s32[2,3]{1,0})->s32[2,3]{1,0}}

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[] parameter(0)
// CHECK-NEXT:  %[[broadcast_4:[^ ]+]] = pred[2,3] broadcast(%[[Arg_0_1]]), dimensions={},
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[2,3] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s32[2,3] parameter(2)
// CHECK-NEXT:  ROOT %[[select_5:[^ ]+]] = s32[2,3] select(%[[broadcast_4]], %[[Arg_1_2]], %[[Arg_2_3]]),
func.func @main(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}
// CHECK-DIRECT: stablehlo.select

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[5,1,5]{2,1,0}, s32[2]{0})->f32[2,1,5]{2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[5,1,5] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[2] parameter(1)
// CHECK-NEXT:  ROOT %[[gather_3:[^ ]+]] = f32[2,1,5] gather(%[[Arg_0_1]], %[[Arg_1_2]]), offset_dims={1,2}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,1,5},
func.func @main(%arg0: tensor<5x1x5xf32>, %arg1: tensor<2xi32>) ->  tensor<2x1x5xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
  dim = 0 : i64,
  batch_dims = 0 : i64
  } : (tensor<5x1x5xf32>, tensor<2xi32>) -> tensor<2x1x5xf32>
  func.return %0 : tensor<2x1x5xf32>
}
// CHECK-DIRECT: stablehlo.torch_index_select

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[1,2,3,4]{3,2,1,0})->s32[2,1,4,3]{2,3,0,1}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[1,2,3,4] parameter(0)
// CHECK-NEXT:  ROOT %[[transpose_2:[^ ]+]] = s32[2,1,4,3] transpose(%[[Arg_0_1]]), dimensions={1,0,3,2},
func.func @main(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0 : tensor<2x1x4x3xi32>
}
// CHECK-DIRECT: stablehlo.transpose

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,4]{1,0}, f32[3,4]{1,0})->f32[3,4]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4,4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[3,4] parameter(1)
// CHECK-NEXT:  ROOT %[[triangular_solve_3:[^ ]+]] = f32[3,4] triangular-solve(%[[Arg_0_1]], %[[Arg_1_2]]), lower=true, transpose_a=NO_TRANSPOSE,
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-DIRECT: stablehlo.triangular_solve

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,4]{1,0}, f32[3,4]{1,0})->f32[3,4]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4,4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[3,4] parameter(1)
// CHECK-NEXT:  ROOT %[[triangular_solve_3:[^ ]+]] = f32[3,4] triangular-solve(%[[Arg_0_1]], %[[Arg_1_2]]), lower=true, transpose_a=NO_TRANSPOSE,
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-DIRECT: stablehlo.triangular_solve

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[4]{0}, pred[4]{0})->pred[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = pred[4] parameter(1)
// CHECK-NEXT:  ROOT %[[xor_3:[^ ]+]] = pred[4] xor(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-DIRECT: stablehlo.xor

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[4]{0}, s32[4]{0})->s32[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[4] parameter(1)
// CHECK-NEXT:  ROOT %[[xor_3:[^ ]+]] = s32[4] xor(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}
// CHECK-DIRECT: stablehlo.xor
