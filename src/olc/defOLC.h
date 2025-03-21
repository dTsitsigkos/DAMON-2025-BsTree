#ifndef _DEFOLC_H_
#define _DEFOLC_H_

#include <iostream> 
#include <set> 
#include <cstring>
#include <math.h>
#include <chrono>
#include <immintrin.h>
#include <bits/stdc++.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <cstdint>

#include <sys/resource.h>
#include <unistd.h>
#include <bits/stdc++.h>


using namespace std;
using namespace std::chrono;

#define STEP_LEAF 3
#define STEP_INNER 7
#define GAPS_IN_LEAF 4
#define GAPS_IN_INNER 1
#define KEYS_IN_LEAF 12
#define KEYS_IN_INNER 14
#define BLOCKSIZE   16
#define ORIGINAL_TREE 0
#define COMPACT_TREE 1


struct OptLock {
    std::atomic<uint64_t> typeVersionLockObsolete{0b100};
  
    bool isLocked(uint64_t version) {
      return ((version & 0b10) == 0b10);
    }
  
    uint64_t readLockOrRestart(bool &needRestart) {
      uint64_t version;
      version = typeVersionLockObsolete.load();
      if (isLocked(version) || isObsolete(version)) {
        _mm_pause();
        needRestart = true;
      }
      return version;
    }
  
    void writeLockOrRestart(bool &needRestart) {
      uint64_t version;
      version = readLockOrRestart(needRestart);
      if (needRestart) return;
  
      upgradeToWriteLockOrRestart(version, needRestart);
      if (needRestart) return;
    }
  
    void upgradeToWriteLockOrRestart(uint64_t &version, bool &needRestart) {
      if (typeVersionLockObsolete.compare_exchange_strong(version, version + 0b10)) {
        version = version + 0b10;
      } else {
        _mm_pause();
        needRestart = true;
      }
    }
  
    void writeUnlock() {
      typeVersionLockObsolete.fetch_add(0b10);
    }
  
    bool isObsolete(uint64_t version) {
      return (version & 1) == 1;
    }
  
    void checkOrRestart(uint64_t startRead, bool &needRestart) const {
      readUnlockOrRestart(startRead, needRestart);
    }
  
    void readUnlockOrRestart(uint64_t startRead, bool &needRestart) const {
      needRestart = (startRead != typeVersionLockObsolete.load());
    }
  
    void writeUnlockObsolete() {
      typeVersionLockObsolete.fetch_add(0b11);
    }
  };


//typedef struct __attribute__((aligned(256)))btree_inner_node{
typedef struct btree_inner_node : public OptLock{
    uint64_t keys[16];
    uint32_t ptrs[16];
} btree_inner;


typedef struct btree_leaf_node : public OptLock{
    uint64_t keys[16];
} btree_leaf;


typedef struct leaf_statistics{
    uint8_t slot_use;
    uint16_t bit_map;
    uint32_t next_leaf;
}stats_leaf;


typedef struct inner_statistics{
    uint8_t slot_use;
    uint16_t bit_map;
}stats_inner;

typedef struct btree_statistics{
    std::atomic <uint32_t> num_leaves;
    std::atomic <uint32_t> max_num_leaves;
    std::atomic <uint32_t> num_inners;
    std::atomic <uint32_t> max_num_inners;
    std::atomic <uint32_t> height;
} btree_stats;

const int page_size = 1 << 21;
std::atomic <uint32_t>root_offset(0);
btree_stats bt_stats;

std::atomic<stats_leaf *>leaves_stats;
std::atomic<stats_inner *> inners_stats;

std::atomic <btree_leaf *>  bt_leaves;
std::atomic <btree_inner *> bt_inners;


uint8_t select_tree_type(uint64_t *ar, uint64_t size_data)
{
    uint64_t diff = 0;
    uint8_t leading_zeros = 0;
    uint64_t avg_leading_zeros = 0;

    uint32_t i = 1;
    for (uint32_t count = KEYS_IN_LEAF; count < size_data;  count += KEYS_IN_LEAF, i++){
        diff = ar[count+KEYS_IN_LEAF]-ar[count];
        leading_zeros = _lzcnt_u64(diff);
        avg_leading_zeros += leading_zeros;
        count += KEYS_IN_LEAF;
    }

    avg_leading_zeros = avg_leading_zeros / i;

    if (avg_leading_zeros >= 32)
    {
        return COMPACT_TREE;
    }
    else
    {
        return ORIGINAL_TREE;
    }
}

#endif