#ifndef _DEF_H_
#define _DEF_H_

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
#define GAPS_IN_LEAF_16BIT 16
#define GAPS_IN_LEAF_32BIT 8
#define GAPS_IN_LEAF_64BIT 4
#define GAPS_IN_INNER 1
#define KEYS_IN_LEAF_64BIT 12
#define KEYS_IN_LEAF_32BIT 24
#define KEYS_IN_LEAF_16BIT 48
#define KEYS_IN_INNER 14
#define BLOCKSIZE_16BIT 64
#define BLOCKSIZE_32BIT 32
#define BLOCKSIZE_64BIT 16
#define ORIGINAL_TREE 0
#define COMPACT_TREE 1


//inner node structure 
typedef struct btree_inner_node{ 
    uint64_t keys[BLOCKSIZE_64BIT];
    uint32_t ptrs[BLOCKSIZE_64BIT];
} btree_inner;


//inner statistics
typedef struct inner_statistics{
    uint8_t slot_use;
    uint16_t bit_map;
}stats_inner;


//leaf node structure - bstree
typedef struct btree_leaf_node{ 
    uint64_t keys[BLOCKSIZE_64BIT];
} btree_leaf;


//leaf statistics - bstree
typedef struct leaf_statistics{ 
    uint8_t slot_use;
    uint16_t bit_map;
    uint32_t next_leaf;
}stats_leaf;


//leaf node structure - compact bstree
typedef struct btree_compact_leaf_node{ 
    uint32_t keys[BLOCKSIZE_32BIT]; 
} btree_compact_leaf;


//leaf statistics - compact bstree
typedef struct compact_leaf_statistics{ 
    uint8_t leaf_case;
    uint8_t slot_use;
    uint32_t next_leaf;
    uint64_t bit_map;
    uint64_t reference_key;
}stats_compact_leaf;


//bstree statistics for all the tree
typedef struct btree_statistics{
    uint32_t num_leaves;
    uint32_t max_num_leaves;
    uint32_t num_inners;
    uint32_t max_num_inners;
    uint32_t height;
} btree_stats;


//global variables
const int page_size = 1 << 21;
uint32_t root_offset = 0;
btree_stats bt_stats;
stats_compact_leaf *comp_leaves_stats;
stats_leaf *leaves_stats;
stats_inner *inners_stats;
btree_inner *bt_inners;


#endif