kernel void sum_block(global uint* data, global uint* block_sums, const uint n) {
  uint global_id = get_global_id(0);
  uint local_id = get_local_id(0);
  uint group_id = get_group_id(0);
  uint tree_level_offset = 1;
  uint local_data_idx = 2 * local_id;
  uint global_data_idx = 2 * global_id;
  local uint shared_data[WORKGROUP * 2];

  shared_data[local_data_idx] = (global_data_idx < n) ? data[global_data_idx] : 0;
  shared_data[local_data_idx + 1] = (global_data_idx + 1 < n) ? data[global_data_idx + 1] : 0;

  for (uint stride = WORKGROUP; stride > 0; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < stride) {
      uint left_child = tree_level_offset * (local_data_idx + 1) - 1;
      uint right_child = tree_level_offset * (local_data_idx + 2) - 1;
      shared_data[right_child] += shared_data[left_child];
    }
    tree_level_offset <<= 1;
  }

  local uint block_sum;
  if (local_id == 0) {
    block_sum = shared_data[WORKGROUP * 2 - 1];
    block_sums[group_id] = block_sum;
    shared_data[WORKGROUP * 2 - 1] = 0;
  }

  for (uint stride = 1; stride < WORKGROUP * 2; stride <<= 1) {
    tree_level_offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < stride) {
      uint left_child = tree_level_offset * (local_data_idx + 1) - 1;
      uint right_child = tree_level_offset * (local_data_idx + 2) - 1;
      uint temp = shared_data[left_child];
      shared_data[left_child] = shared_data[right_child];
      shared_data[right_child] += temp;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (global_data_idx < n) {
    data[global_data_idx] = shared_data[local_data_idx + 1];
  }
  if (global_data_idx + 1 < n) {
    data[global_data_idx + 1] = local_data_idx + 2 < WORKGROUP * 2 ? shared_data[local_data_idx + 2] : block_sum;
  }
}

kernel void sum_add(global uint* data, global uint* block_sums, const uint n) {
  uint global_id = get_global_id(0);
  uint local_id = get_local_id(0);
  uint group_id = get_group_id(0);

  local uint previous_block_sum;
  if (local_id == 0) {
    previous_block_sum = (group_id > 0) ? block_sums[group_id - 1] : 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint global_data_idx = 2 * global_id;
  if (global_data_idx < n) {
    data[global_data_idx] += previous_block_sum;
  }
  if (global_data_idx + 1 < n) {
    data[global_data_idx + 1] += previous_block_sum;
  }
}
