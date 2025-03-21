#ifndef _COMPACT_BSTREE_H_
#define _COMPACT_BSTREE_H_

#include <bits/stdc++.h>
#include <immintrin.h>
#include "def.h"

//global variables about the leaves 
btree_compact_leaf *bt_comp_leaves;

//print leaves nodes
void print_leaves_compact(){
    uint16_t temp_arr_16bit[BLOCKSIZE_16BIT];
    uint64_t temp_arr_64bit[BLOCKSIZE_64BIT];

    for (uint32_t i = 0; i < bt_stats.num_leaves; i++){
        printf("For Leaf Node %u: \n", i);
        uint8_t leaf_case = comp_leaves_stats[i].leaf_case;
        switch(leaf_case){
            case 0:
                printf("Compressed Information with 16-bit Differences from a Reference Key %lu\n", comp_leaves_stats[i].reference_key);
                printf("- Keys:\n");
                memcpy(&temp_arr_16bit, &bt_comp_leaves[i].keys, BLOCKSIZE_16BIT * sizeof(uint16_t));
                for (uint32_t j = 0; j < BLOCKSIZE_16BIT; j++){
                    printf("%u ", temp_arr_16bit[j]);
                }
                break;
            case 1:
                printf("Compressed Information with 32-bit Differences from a Reference Key %lu\n", comp_leaves_stats[i].reference_key);
                printf("- Keys:\n");
                
                for (uint32_t j = 0; j < BLOCKSIZE_32BIT; j++){
                    printf("%u ", bt_comp_leaves[i].keys[j]);
                }
                break;

            case 2:
                printf("Exact Information with 64-bit key values\n");
                printf("- Keys:\n");
                memcpy(&temp_arr_64bit, &bt_comp_leaves[i].keys, BLOCKSIZE_64BIT * sizeof(uint64_t));
                for (uint32_t j = 0; j < BLOCKSIZE_64BIT; j++){
                    printf("%lu ", temp_arr_64bit[j]);
                }
                break;
        }

        printf("\n- Statistics:\n");
        printf("Slotuse = %d -- Bitmap = 0x%lX -- nextLeaf = %u\n\n", comp_leaves_stats[i].slot_use, comp_leaves_stats[i].bit_map, comp_leaves_stats[i].next_leaf);
    }
}


//print inner nodes
void print_inners_compact(){
    for (uint32_t i = 0; i < bt_stats.num_inners; i++){   
        printf("For Inner Node %u: \n", i);
        printf("- Keys:\n");
        for (uint32_t j = 0; j < BLOCKSIZE_64BIT; j++){
            printf("%lu ", bt_inners[i].keys[j]);
        }
        
        printf("\n- Pointers:\n");
        for (uint32_t j = 0; j < BLOCKSIZE_64BIT; j++){
            printf("%u ", bt_inners[i].ptrs[j]);
        }

        printf("\n- Statistics:\n");
        printf("Slotuse = %d -- Bitmap = 0x%X\n\n", inners_stats[i].slot_use, inners_stats[i].bit_map);
    }
}


//reallocation in leaves 
void reallocate_leaves_compact() {
    btree_compact_leaf *old_ptr = bt_comp_leaves;
    btree_compact_leaf *new_ptr;
    stats_compact_leaf *old_stats = comp_leaves_stats, *new_stats;

    //increase the number of leaves
    bt_stats.max_num_leaves = 2 * bt_stats.num_leaves;

    //allocate new array for leaves
    new_ptr = (btree_compact_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(btree_compact_leaf));
    
    if (new_ptr == NULL){
        printf("BAD ALLOCATION AT LEAF NODES\n");
        exit(-1);
    }

    //allocate new array for leaves stats
    new_stats = (stats_compact_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(stats_compact_leaf));

    if (new_stats == NULL){
        printf("BAD ALLOCATION AT LEAF STATS\n");
        exit(-1);
    }

    //copy the previous array of leaves to the new ones
    memcpy(new_ptr, old_ptr, bt_stats.num_leaves * sizeof(btree_compact_leaf));
    bt_comp_leaves = new_ptr;

     //copy the previous array of leaves stats to the new ones
    memcpy(new_stats, old_stats, bt_stats.num_leaves * sizeof(stats_compact_leaf));
    comp_leaves_stats = new_stats;

    //free the previous arrays
    free(old_ptr);
    free(old_stats);
}


// reallocation in inners
void reallocate_inners_compact(){
    btree_inner *old_ptr = bt_inners;
    btree_inner *new_ptr;
    stats_inner *old_stats = inners_stats, *new_stats;    

    //increase the number of inners
    bt_stats.max_num_inners =  2* bt_stats.max_num_inners; //initialize max num inner nodes

    //allocate new array for innert using huge pages
    uint32_t num_pages_inner = ((bt_stats.max_num_inners * sizeof(btree_inner) -1) / page_size) + 1;//initialize num pages for inner nodes
    new_ptr = (btree_inner *) aligned_alloc(page_size, num_pages_inner * page_size);
    madvise(bt_inners, num_pages_inner*page_size, MADV_HUGEPAGE);

    //allocate new array for inner stats
    if (new_ptr == NULL){
        printf("BAD ALLOCATION AT INNER NODES\n");
        exit(-1);
    }

    //copy the previous array of inners to the new ones
    new_stats = (stats_inner *) aligned_alloc(64, bt_stats.max_num_inners*sizeof(stats_inner));

    if (new_stats == NULL){
        printf("BAD ALLOCATION AT INNER STATS\n");
        exit(-1);
    }

    //copy the previous array of inners stats to the new ones
    memcpy(new_ptr, old_ptr, bt_stats.num_inners * sizeof(btree_inner));
    bt_inners = new_ptr;

    memcpy(new_stats, old_stats, bt_stats.num_inners * sizeof(stats_inner));
    inners_stats = new_stats;

    //free the previous arrays
    free(old_ptr);
    free(old_stats);

}


// finds lowest position greater than key with count
inline uint32_t successorLinG_compact(uint64_t *ar, uint64_t key) {
    uint32_t s1 = 0;
    
    __m512i threshold_vector = _mm512_set1_epi64(key); //load the search key into one register
    
    __m512i y1 = _mm512_loadu_epi64((__m512i*) ar);// load the first 8 keys from the node into one register 
    uint32_t m1 = (uint32_t) _mm512_cmpge_epu64_mask(threshold_vector,y1);//compare threshold_vector >= y1
    s1 = _mm_popcnt_u32(m1);//pop count the result of the comparison

    __m512i y2 = _mm512_loadu_epi64((__m512i*)(ar + 8)); // load the last 8 keys from the node into one register 
    uint32_t m2 = (uint32_t) _mm512_cmpge_epu64_mask(threshold_vector,y2); //compare threshold_vector >= y2
    s1 += _mm_popcnt_u32(m2);//pop count the result of the comparison and add it with the previous one

    return s1;
}


// finds lowest position greater than or equal to key with count for a leaf node including 16-bit differences
uint32_t successorLinGE_compact(void * ar, uint64_t previousLevelKey, uint8_t leaf_case, uint64_t *skey){
    uint32_t s1 = 0;
    uint32_t m1, m2;
    __m512i y1, y2;
    __m512i threshold_vector;

    switch(leaf_case){
        case 0:{ // 16 bit keys
            uint16_t * array = (uint16_t *) ar;
            uint16_t key = (uint16_t) (*skey - previousLevelKey);

            threshold_vector = _mm512_set1_epi16(key);
            
            y1 = _mm512_loadu_epi16((__m512*) array);
            m1 = (uint32_t) _mm512_cmpgt_epu16_mask(threshold_vector, y1);
            s1 = _mm_popcnt_u32(m1);

            y2 = _mm512_loadu_epi16((__m512*) (array + 32));
            m2 = (uint32_t) _mm512_cmpgt_epu16_mask(threshold_vector, y2);
            s1 +=  _mm_popcnt_u32(m2);

            s1 = (s1 == BLOCKSIZE_16BIT) ? --s1 : s1;

            *skey = (uint16_t) key;    
            break;    
        }
        case 1:{//32 bit keys
            uint32_t *array = (uint32_t *) ar;
            uint32_t key = (uint32_t) (*skey - previousLevelKey);

            threshold_vector = _mm512_set1_epi32(key);

            y1 = _mm512_loadu_epi32((__m512i*) array);
            m1 = (uint32_t) _mm512_cmpgt_epu32_mask(threshold_vector, y1);
            s1 = _mm_popcnt_u32(m1);

            y2 = _mm512_loadu_epi32((__m512i*) (array + 16));
            m2 = (uint32_t) _mm512_cmpgt_epu32_mask(threshold_vector, y2);
            s1 += _mm_popcnt_u32(m2);

            s1 = (s1 == BLOCKSIZE_32BIT) ? --s1 : s1;

            *skey = (uint32_t) key;
            break;    
        }
        case 2:{//64 bit keys
            uint64_t *array = (uint64_t *) ar;

            threshold_vector = _mm512_set1_epi64(*skey);
            
            y1 = _mm512_loadu_epi64((__m512i*) array);
            m1 = (uint32_t) _mm512_cmpgt_epu64_mask(threshold_vector, y1);
            s1 = _mm_popcnt_u32(m1);

            y2 = _mm512_loadu_epi64((__m512i*) (array + 8)); 
            m2 = (uint32_t) _mm512_cmpgt_epu64_mask(threshold_vector, y2);
            s1 += _mm_popcnt_u32(m2);

            s1 = (s1 == BLOCKSIZE_64BIT) ? --s1: s1;
        break;        
        }
    }

    return s1;
}


//construct the leaves nodes of the compact bstree, 
//same code with original bstree, but we have variable length in leaves
uint64_t *construct_leaves_compact(uint64_t *ar, uint64_t size_data){   
    uint8_t leaf_case, leading_zeros = 0;
    uint32_t i = 0, j = 0, created_leaves = 0;
    uint64_t limited_size = 3 << 30;//limit of transparent huge pages
    uint64_t NUMnextlevelnonleafkeys = 0, prev_inserted_leaf = 0;    
    uint64_t min_key_at_leaf, bitmap, diff;
    uint64_t *nextlevelnonleafkeys;
    uint16_t temp_array_for_16bit_keys[BLOCKSIZE_16BIT] = {0};
    uint64_t temp_array_for_64bit_keys[BLOCKSIZE_64BIT] = {0};

    //allocate leaves nodes and leaves statistics array
    bt_stats.num_leaves = ceil((double)size_data/KEYS_IN_LEAF_64BIT);   //initialize num leaves based on the num of gaps  
    bt_stats.max_num_leaves = 2 * bt_stats.num_leaves; //initialize max num leaves

    if (bt_stats.max_num_leaves * sizeof(btree_compact_leaf) <= limited_size){// leaves nodes into huge pages
        uint32_t num_pages_leaves = ((bt_stats.max_num_leaves * sizeof(btree_compact_leaf) -1) / page_size) + 1;

        bt_comp_leaves = (btree_compact_leaf *) aligned_alloc(page_size,  num_pages_leaves*page_size);//allocation memory for leaves
        madvise(bt_comp_leaves, num_pages_leaves * page_size, MADV_HUGEPAGE);
    }
    else{//leaves nodes not into huge pages
        bt_comp_leaves = (btree_compact_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(btree_compact_leaf));

    }
    
    uint32_t num_pages_for_stats = ((bt_stats.max_num_leaves * sizeof(stats_compact_leaf) -1) / page_size);
    comp_leaves_stats = (stats_compact_leaf *) aligned_alloc(page_size, num_pages_for_stats * page_size);   // allocate the additional space for the statistics of the leaves
    madvise(comp_leaves_stats, num_pages_for_stats * page_size, MADV_HUGEPAGE);

    nextlevelnonleafkeys = (uint64_t *)aligned_alloc(64, sizeof(uint64_t) * bt_stats.num_leaves);   // keys (separators) stored in non-leaf nodes (next level)

    // Create the first leaf to include exact keys (64-bit)
    min_key_at_leaf = ar[0];
    diff = ar[BLOCKSIZE_64BIT - 1] - min_key_at_leaf;

    if (diff == BLOCKSIZE_64BIT - 1){//sequential keys
        memcpy(&bt_comp_leaves[0].keys, ar, BLOCKSIZE_64BIT * sizeof(uint64_t));
        prev_inserted_leaf += BLOCKSIZE_64BIT;
        created_leaves++;

        comp_leaves_stats[0].leaf_case = 2;
        comp_leaves_stats[0].slot_use = BLOCKSIZE_64BIT;
        comp_leaves_stats[0].next_leaf = created_leaves;
        comp_leaves_stats[0].bit_map = 0xffff;
        comp_leaves_stats[0].reference_key = min_key_at_leaf;
    }
    else{//non-sequential keys
        for (j = 0; j < BLOCKSIZE_64BIT - 4; j += 4){
            temp_array_for_64bit_keys[j] = ar[prev_inserted_leaf];
            prev_inserted_leaf++;

            temp_array_for_64bit_keys[j + 1] = ar[prev_inserted_leaf];
            prev_inserted_leaf++;

            temp_array_for_64bit_keys[j + 2] = ar[prev_inserted_leaf];
            prev_inserted_leaf++;

            temp_array_for_64bit_keys[j + 3] = ar[prev_inserted_leaf];
        }

        temp_array_for_64bit_keys[j] = ar[prev_inserted_leaf];
        prev_inserted_leaf++;

        temp_array_for_64bit_keys[j+1] = ar[prev_inserted_leaf];
        prev_inserted_leaf++;

        temp_array_for_64bit_keys[j+2] = ar[prev_inserted_leaf];
        prev_inserted_leaf++;

        temp_array_for_64bit_keys[j+3] = ULONG_MAX;

        memcpy(&bt_comp_leaves[0].keys, &temp_array_for_64bit_keys, BLOCKSIZE_64BIT * sizeof(uint64_t));

        created_leaves++;

        comp_leaves_stats[0].leaf_case = 2;
        comp_leaves_stats[0].slot_use = KEYS_IN_LEAF_64BIT;
        comp_leaves_stats[0].next_leaf = created_leaves;
        comp_leaves_stats[0].bit_map = 0xeeee;
        comp_leaves_stats[0].reference_key = min_key_at_leaf;
    }

    //create the other leafs except of the last 16-bit node
    for (; prev_inserted_leaf + KEYS_IN_LEAF_16BIT < size_data; created_leaves++){
        uint32_t high = BLOCKSIZE_16BIT - 1;
        uint32_t ancestor = BLOCKSIZE_16BIT;
        uint32_t threshold = 48;
        uint32_t gaps = 16;
        uint8_t sequential_case = 0;
        leaf_case = 0;
        leading_zeros = 0;

        min_key_at_leaf = ar[prev_inserted_leaf];

        while (leaf_case < 2){
            diff = ar[prev_inserted_leaf + high] - min_key_at_leaf;

            if (diff == high){
                sequential_case = 1;
                diff = ar[prev_inserted_leaf + ancestor] - min_key_at_leaf;
            }
            else{
                uint32_t temp_ancestor = ancestor - gaps;
                diff = ar[prev_inserted_leaf + temp_ancestor] - min_key_at_leaf;
            }

            leading_zeros = _lzcnt_u64(diff);

            if (leading_zeros >= threshold){
                leaf_case = (sequential_case << 4) | leaf_case;
                break;
            }

            high = high / 2;
            ancestor = ancestor / 2;
            threshold = threshold - gaps;
            gaps = gaps / 2;
            leaf_case++;
        }

        nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = min_key_at_leaf;
        comp_leaves_stats[created_leaves].reference_key = min_key_at_leaf;
        switch (leaf_case){
            case 0x00: // non-sequntial 16 bit differences
                
                comp_leaves_stats[created_leaves].leaf_case = 0;
                comp_leaves_stats[created_leaves].slot_use = KEYS_IN_LEAF_16BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xeeeeeeeeeeeeeeee;

                for (j = 0; j < BLOCKSIZE_16BIT - 4; j += 4){
                    temp_array_for_16bit_keys[j] = (uint16_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    temp_array_for_16bit_keys[j + 1] = (uint16_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    temp_array_for_16bit_keys[j + 2] = (uint16_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    temp_array_for_16bit_keys[j + 3] = (uint16_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                }

                temp_array_for_16bit_keys[j] = (uint16_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                temp_array_for_16bit_keys[j + 1] = (uint16_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                temp_array_for_16bit_keys[j + 2] = (uint16_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                temp_array_for_16bit_keys[j + 3] = USHRT_MAX;
                
                memcpy(&bt_comp_leaves[created_leaves].keys, &temp_array_for_16bit_keys, BLOCKSIZE_16BIT * sizeof(uint16_t));

                break;

            case 0x10: //sequential 16-bit differences

                comp_leaves_stats[created_leaves].leaf_case = 0;
                comp_leaves_stats[created_leaves].slot_use = BLOCKSIZE_16BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xffffffffffffffff;

                for (j = 0; j < BLOCKSIZE_16BIT; j++){
                    temp_array_for_16bit_keys[j] = (uint16_t) j;
                }

                memcpy(&bt_comp_leaves[created_leaves].keys, &temp_array_for_16bit_keys, BLOCKSIZE_16BIT * sizeof(uint16_t));
                prev_inserted_leaf += BLOCKSIZE_16BIT;
                
                break;

            case 0x01:// non-sequential 32-bit differences
                
                comp_leaves_stats[created_leaves].leaf_case = 1;
                comp_leaves_stats[created_leaves].slot_use = KEYS_IN_LEAF_32BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xeeeeeeee;

                for (j = 0; j < BLOCKSIZE_32BIT - 4; j += 4){
                    bt_comp_leaves[created_leaves].keys[j] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    bt_comp_leaves[created_leaves].keys[j + 1] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    bt_comp_leaves[created_leaves].keys[j + 2] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    bt_comp_leaves[created_leaves].keys[j + 3] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                }

                bt_comp_leaves[created_leaves].keys[j] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                bt_comp_leaves[created_leaves].keys[j + 1] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                bt_comp_leaves[created_leaves].keys[j + 2] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                bt_comp_leaves[created_leaves].keys[j + 3] = UINT_MAX;

                break;

            case 0x11: //sequential 32-bit differences

                comp_leaves_stats[created_leaves].leaf_case = 1;
                comp_leaves_stats[created_leaves].slot_use = BLOCKSIZE_32BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xffffffff;

                for (j = 0; j < BLOCKSIZE_32BIT; j++){
                    bt_comp_leaves[created_leaves].keys[j] = (uint32_t) j;
                }

                prev_inserted_leaf += BLOCKSIZE_32BIT;
                
                break;

            case 0x02://non=sequential 64-bit keys
                
                comp_leaves_stats[created_leaves].leaf_case = 2;
                comp_leaves_stats[created_leaves].slot_use = KEYS_IN_LEAF_64BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xeeee;

                for (j = 0; j < BLOCKSIZE_64BIT - 4; j += 4){
                    temp_array_for_64bit_keys[j] = ar[prev_inserted_leaf];
                    prev_inserted_leaf++;

                    temp_array_for_64bit_keys[j + 1] = ar[prev_inserted_leaf];
                    prev_inserted_leaf++;
                    
                    temp_array_for_64bit_keys[j + 2] = ar[prev_inserted_leaf];
                    prev_inserted_leaf++;
                    
                    temp_array_for_64bit_keys[j + 3] = ar[prev_inserted_leaf];
                }

                temp_array_for_64bit_keys[j] = ar[prev_inserted_leaf];
                prev_inserted_leaf++;

                temp_array_for_64bit_keys[j + 1] = ar[prev_inserted_leaf];
                prev_inserted_leaf++;
                
                temp_array_for_64bit_keys[j + 2] = ar[prev_inserted_leaf];
                prev_inserted_leaf++;

                temp_array_for_64bit_keys[j + 3] = ULONG_MAX;

                memcpy(&bt_comp_leaves[created_leaves].keys, &temp_array_for_64bit_keys, BLOCKSIZE_64BIT * sizeof(uint64_t));

                break;

            case 0x12: //sequential 64-bit keys
                
                comp_leaves_stats[created_leaves].leaf_case = 2;
                comp_leaves_stats[created_leaves].slot_use = BLOCKSIZE_64BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xffff;

                memcpy(&bt_comp_leaves[created_leaves].keys, &ar[prev_inserted_leaf], BLOCKSIZE_64BIT * sizeof(uint64_t));
                prev_inserted_leaf += BLOCKSIZE_64BIT;

                break;
        }
    }

    // Find the number and the type of the remaining leaves
    int residue = size_data - prev_inserted_leaf;    
    vector < pair <uint32_t, uint8_t> > remaining_leaves;
    uint64_t temp_prev_inserted_leaf = prev_inserted_leaf;

    while (1){
        min_key_at_leaf = ar[temp_prev_inserted_leaf];

        diff = ar[temp_prev_inserted_leaf + residue - 1] - min_key_at_leaf;
        leading_zeros = _lzcnt_u64(diff);

        if (residue > BLOCKSIZE_32BIT){
            diff = ar[temp_prev_inserted_leaf + BLOCKSIZE_32BIT - 1] - min_key_at_leaf;

            if (diff == BLOCKSIZE_32BIT - 1){
                diff = ar[temp_prev_inserted_leaf + BLOCKSIZE_32BIT] - min_key_at_leaf;
                leading_zeros = _lzcnt_u64(diff);

                if (leading_zeros >= 32){
                    remaining_leaves.push_back(make_pair(BLOCKSIZE_32BIT, 0x11));
                    temp_prev_inserted_leaf += BLOCKSIZE_32BIT;
                    residue -= BLOCKSIZE_32BIT;
                }
                else{
                    remaining_leaves.push_back(make_pair(BLOCKSIZE_64BIT, 0x12));
                    temp_prev_inserted_leaf += BLOCKSIZE_64BIT;
                    residue -= BLOCKSIZE_64BIT;
                }
            }
            else{
                diff = ar[temp_prev_inserted_leaf + KEYS_IN_LEAF_32BIT] - min_key_at_leaf;
                leading_zeros = _lzcnt_u64(diff);
            
                if (leading_zeros >= 32){
                    remaining_leaves.push_back(make_pair(KEYS_IN_LEAF_32BIT, 0x01));
                    temp_prev_inserted_leaf += KEYS_IN_LEAF_32BIT;
                    residue -= KEYS_IN_LEAF_32BIT;
                }
                else{
                    diff = ar[temp_prev_inserted_leaf + BLOCKSIZE_64BIT - 1] - min_key_at_leaf;

                    if (diff == BLOCKSIZE_64BIT - 1){
                        remaining_leaves.push_back(make_pair(BLOCKSIZE_64BIT, 0x12));
                        temp_prev_inserted_leaf += BLOCKSIZE_64BIT;
                        residue -= BLOCKSIZE_64BIT;
                    }
                    else{
                        remaining_leaves.push_back(make_pair(KEYS_IN_LEAF_64BIT, 0x02));
                        temp_prev_inserted_leaf += KEYS_IN_LEAF_64BIT;
                        residue -= KEYS_IN_LEAF_64BIT;
                    }
                }
            }

            continue;
        }

        if (residue <= BLOCKSIZE_64BIT){
            remaining_leaves.push_back(make_pair(residue, 0x02));
            break;
        }
        else{
            diff = ar[temp_prev_inserted_leaf + BLOCKSIZE_64BIT - 1] - min_key_at_leaf;

            if (diff == BLOCKSIZE_64BIT - 1){
                remaining_leaves.push_back(make_pair(BLOCKSIZE_64BIT, 0x12));
                temp_prev_inserted_leaf += BLOCKSIZE_64BIT;
                residue -= BLOCKSIZE_64BIT;
            }
            else{
                remaining_leaves.push_back(make_pair(KEYS_IN_LEAF_64BIT, 0x02));
                temp_prev_inserted_leaf += KEYS_IN_LEAF_64BIT;
                residue -= KEYS_IN_LEAF_64BIT;
            }

            continue;
        }
    }
    
    //the node before the last node can not be with 16-bit differences, because the last node should be with 64-bit keys
    for (i = 0; i < remaining_leaves.size() - 1; i++, created_leaves++){
        min_key_at_leaf = ar[prev_inserted_leaf];
        nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = min_key_at_leaf;
        comp_leaves_stats[created_leaves].reference_key = min_key_at_leaf;
        switch (remaining_leaves[i].second){
            case 0x01: //non-sequential 32-bit differences
                comp_leaves_stats[created_leaves].leaf_case = 1;
                comp_leaves_stats[created_leaves].slot_use = KEYS_IN_LEAF_32BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xeeeeeeee;
                
                for (j = 0; j < BLOCKSIZE_32BIT - 4; j += 4){
                    bt_comp_leaves[created_leaves].keys[j] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    bt_comp_leaves[created_leaves].keys[j + 1] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    bt_comp_leaves[created_leaves].keys[j + 2] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                    prev_inserted_leaf++;

                    bt_comp_leaves[created_leaves].keys[j + 3] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                }

                bt_comp_leaves[created_leaves].keys[j] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                bt_comp_leaves[created_leaves].keys[j + 1] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                bt_comp_leaves[created_leaves].keys[j + 2] = (uint32_t) (ar[prev_inserted_leaf] - min_key_at_leaf);
                prev_inserted_leaf++;

                bt_comp_leaves[created_leaves].keys[j + 3] = UINT_MAX;

                break;

            case 0x11: //sequential 32-bit differences

                comp_leaves_stats[created_leaves].leaf_case = 1;
                comp_leaves_stats[created_leaves].slot_use = BLOCKSIZE_32BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xffffffff;

                for (j = 0; j < BLOCKSIZE_32BIT; j++){
                    bt_comp_leaves[created_leaves].keys[j] = (uint32_t) j;
                }

                prev_inserted_leaf += BLOCKSIZE_32BIT;

                break;

            case 0x02:// non-sequential 64-bit keys
                
                comp_leaves_stats[created_leaves].leaf_case = 2;
                comp_leaves_stats[created_leaves].slot_use = KEYS_IN_LEAF_64BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xeeee;

                for (j = 0; j < BLOCKSIZE_64BIT - 4; j += 4){
                    temp_array_for_64bit_keys[j] = ar[prev_inserted_leaf];
                    prev_inserted_leaf++;

                    temp_array_for_64bit_keys[j + 1] = ar[prev_inserted_leaf];
                    prev_inserted_leaf++;
                    
                    temp_array_for_64bit_keys[j + 2] = ar[prev_inserted_leaf];
                    prev_inserted_leaf++;
                    
                    temp_array_for_64bit_keys[j + 3] = ar[prev_inserted_leaf];
                }

                temp_array_for_64bit_keys[j] = ar[prev_inserted_leaf];
                prev_inserted_leaf++;

                temp_array_for_64bit_keys[j + 1] = ar[prev_inserted_leaf];
                prev_inserted_leaf++;
                
                temp_array_for_64bit_keys[j + 2] = ar[prev_inserted_leaf];
                prev_inserted_leaf++;

                temp_array_for_64bit_keys[j + 3] = ULONG_MAX;

                memcpy(&bt_comp_leaves[created_leaves].keys, &temp_array_for_64bit_keys, BLOCKSIZE_64BIT * sizeof(uint64_t));

                break;
            case 0x12: //sequential 64-bit keys

                comp_leaves_stats[created_leaves].leaf_case = 2;
                comp_leaves_stats[created_leaves].slot_use = BLOCKSIZE_64BIT;
                comp_leaves_stats[created_leaves].next_leaf = created_leaves + 1;
                comp_leaves_stats[created_leaves].bit_map = 0xffff;
                
                memcpy(&bt_comp_leaves[created_leaves].keys, &ar[prev_inserted_leaf], BLOCKSIZE_64BIT * sizeof(uint64_t));
                prev_inserted_leaf += BLOCKSIZE_64BIT;

                break;
        }
    }

    //Create the last leaf node with 64-bit keys to avoid issues with inserting large keys.
    min_key_at_leaf = ar[prev_inserted_leaf];
    nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = min_key_at_leaf;

    uint8_t slot_use = remaining_leaves[i].first;
        
    comp_leaves_stats[created_leaves].leaf_case = 2;
    comp_leaves_stats[created_leaves].slot_use = slot_use;
    comp_leaves_stats[created_leaves].next_leaf = 0;
    comp_leaves_stats[created_leaves].bit_map = ((0xffff << (BLOCKSIZE_64BIT - slot_use)) & 0x000000000000ffff);
    comp_leaves_stats[created_leaves].reference_key = min_key_at_leaf;

    for (j = 0; j < slot_use; j++){
        temp_array_for_64bit_keys[j] = ar[prev_inserted_leaf];
        prev_inserted_leaf++;
    }

    for (; j < BLOCKSIZE_64BIT; j++){
        temp_array_for_64bit_keys[j] = ULONG_MAX;
    }


    memcpy(&bt_comp_leaves[created_leaves].keys, &temp_array_for_64bit_keys, BLOCKSIZE_64BIT * sizeof(uint64_t));

    bt_stats.num_leaves = ++created_leaves;

    return nextlevelnonleafkeys;
}


//construct the inner nodes of the compact bstree
void construct_inners_compact(uint64_t *nextlevelnonleafkeys){
    uint64_t i = 0, j = 0;
    uint64_t NUMnextlevelnonleafkeys = bt_stats.num_leaves;
    uint64_t prev_inserted_inner = 0;
    uint32_t nextpointedblockid = 0; 
    uint32_t previous_level = 0;
    uint32_t numcblocks =  0; 
    uint32_t idx = 0;

    uint32_t levels_without_gaps = 1;
    uint32_t height_16 = ceil(log2(bt_stats.num_leaves)/log2(BLOCKSIZE_64BIT));
    uint32_t height_15 = ceil(log2(bt_stats.num_leaves)/log2(KEYS_IN_INNER + 1));
    
    //height optimized mechanism
    if (height_16 == height_15){
        bt_stats.height = height_15;
    }
    else{
        bt_stats.height = height_16;
        levels_without_gaps = floor(height_15/2);
    }

    //create and set values to the lut (lookup table)
    uint32_t lut[bt_stats.height]; // lookup table initialization 
    for (i = 0; i < bt_stats.height-levels_without_gaps; i++){ // put gaps to the specific level (based on optimized mechanism)
        lut[i] = ceil((double)NUMnextlevelnonleafkeys/(KEYS_IN_INNER + 1));
        bt_stats.num_inners += lut[i];
        NUMnextlevelnonleafkeys = lut[i];
    }

    for (; i < bt_stats.height-1; i++){// in the remaining levels does not insert gaps (based on optimized mechanism)
        lut[i] = ceil((double)NUMnextlevelnonleafkeys/(BLOCKSIZE_64BIT));
        bt_stats.num_inners += lut[i];
        NUMnextlevelnonleafkeys = lut[i];
    }

    lut[bt_stats.height - 1] = 1; // root always has one node
    bt_stats.num_inners += 1;

    //allocate memory for inner nodes using huge pages
    bt_stats.max_num_inners =  bt_stats.max_num_leaves / 10; //initialize max num inner nodes
    uint32_t num_pages_inner = ((bt_stats.max_num_inners * sizeof(btree_inner) -1) / page_size) + 1;//initialize num pages for inner nodes
    bt_inners = (btree_inner *)aligned_alloc(page_size,  num_pages_inner*page_size); //allocation memory for inner
    madvise(bt_inners, num_pages_inner*page_size, MADV_HUGEPAGE);

    //allocate memory for inner stats
    inners_stats  = (stats_inner *) aligned_alloc(64, bt_stats.max_num_inners * sizeof(stats_inner)); //allocate additional space for the statistics of the inners

    //create all the levels of inners
    NUMnextlevelnonleafkeys = bt_stats.num_leaves-1;
    if (NUMnextlevelnonleafkeys){	
        uint32_t level = 0;
        for (; level < bt_stats.height-levels_without_gaps; level ++){
            prev_inserted_inner = 0;
			
            uint32_t numptrs = NUMnextlevelnonleafkeys + 1; // include leftmost pointer which has no key    
            NUMnextlevelnonleafkeys = 0;
					
            numcblocks = lut[level];
            idx = 0;
            for (j = level + 1; j < bt_stats.height ; j ++){// calculate the position that each level will be saved in the array of inner nodes. bottom-up implementation
                idx += lut[j];// idx is used to write inner nodes to the appropriate position to the inner array from right to left(first left position is root)
            }
			
            //create inner nodes with gaps (base on the height optimized mechanism)
            for (i=0; i<numcblocks-1; i++) {

                inners_stats[idx+i].slot_use = KEYS_IN_INNER;
                inners_stats[idx+i].bit_map = 0xfeff;

                //insert the first seven keys, before the gap
                memcpy(&bt_inners[idx + i].keys[0],&nextlevelnonleafkeys[prev_inserted_inner], STEP_INNER*sizeof(uint64_t));

                for (ushort t=0; t < STEP_INNER ; t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += STEP_INNER;
                
                //insert the gap
                bt_inners[idx + i].keys[STEP_INNER] = nextlevelnonleafkeys[prev_inserted_inner];
                bt_inners[idx + i].ptrs[STEP_INNER] = nextpointedblockid;
                    
                
                //insert the last seven keys, after the gap
                memcpy(&bt_inners[idx + i].keys[STEP_INNER + 1],&nextlevelnonleafkeys[prev_inserted_inner], STEP_INNER*sizeof(uint64_t));
                for (ushort t= STEP_INNER + 1; t<=BLOCKSIZE_64BIT -1;t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += STEP_INNER;
                
                bt_inners[idx + i].keys[BLOCKSIZE_64BIT-1] = ULONG_MAX;

                //update parent array for the next level
                nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = nextlevelnonleafkeys[prev_inserted_inner];
                prev_inserted_inner ++;          
			}

            uint32_t residue = numptrs%(KEYS_IN_INNER+1);
            uint16_t temp_bit_map = 0xffff;
        
            //Create last inner node per level
            //if residue equals to zero, then we have the same process as the previous nodes
            if (residue==0){
                inners_stats[idx+i].slot_use = KEYS_IN_INNER;
                inners_stats[idx+i].bit_map = 0xfeff;

                //insert the first seven keys, before the gap
                memcpy(&bt_inners[idx + i].keys[0],&nextlevelnonleafkeys[prev_inserted_inner], STEP_INNER*sizeof(uint64_t));
                for (ushort t=0; t < STEP_INNER ; t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += STEP_INNER;
                
                //insert the gap
                bt_inners[idx + i].keys[STEP_INNER] = nextlevelnonleafkeys[prev_inserted_inner];
                bt_inners[idx + i].ptrs[STEP_INNER] = nextpointedblockid;    
                
                //insert the last seven keys, after the gap
                memcpy(&bt_inners[idx + i].keys[STEP_INNER + 1],&nextlevelnonleafkeys[prev_inserted_inner], STEP_INNER*sizeof(uint64_t));
                for (ushort t= STEP_INNER + 1; t<=BLOCKSIZE_64BIT -1;t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += STEP_INNER;
                
                bt_inners[idx + i].keys[BLOCKSIZE_64BIT-1] = ULONG_MAX;

            } 
            // if residue is greater than step inner we calclulate the new step inner and we follow the above process 
            else if (residue > STEP_INNER){

                uint32_t step_inner = residue/(GAPS_IN_INNER + 1);

                uint16_t flip_bit_mask = 0x1;

                //insert step inner keys, before the gap
                memcpy(&bt_inners[idx + i].keys[0],&nextlevelnonleafkeys[prev_inserted_inner],step_inner*sizeof(uint64_t));
                for (ushort t=0; t<step_inner;t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += step_inner;
                
                //insert the gap
                bt_inners[idx + i].keys[step_inner] = nextlevelnonleafkeys[prev_inserted_inner];
                bt_inners[idx + i].ptrs[step_inner] = nextpointedblockid;

                flip_bit_mask = flip_bit_mask <<(16 - step_inner - 1);
                temp_bit_map = temp_bit_map ^ flip_bit_mask;

                uint32_t remaining = residue - 1 - step_inner;

                //insert the remaining keys
                memcpy(&bt_inners[idx + i].keys[step_inner + 1],&nextlevelnonleafkeys[prev_inserted_inner],remaining*sizeof(uint64_t));
                for (ushort t = step_inner + 1; t <= residue;t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += remaining;
                

                //insert the available gaps and update the bit map
                flip_bit_mask = flip_bit_mask >> (remaining + 1);
                nextpointedblockid--;
                for (ushort t = residue; t < BLOCKSIZE_64BIT; t++){
                    bt_inners[idx + i].keys[t] = ULONG_MAX;
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid ;                    

                    if (t < BLOCKSIZE_64BIT - 1){
                        temp_bit_map = temp_bit_map ^ flip_bit_mask;
                        flip_bit_mask = flip_bit_mask >> 1;
                    }
                }
                inners_stats[idx+i].slot_use = residue-1;
                inners_stats[idx+i].bit_map = temp_bit_map;
            }
            //In any other case, we insert the remaining keys and then we insert the gaps
            else{
                temp_bit_map = temp_bit_map << ( 16 - residue +1);

                temp_bit_map = temp_bit_map | 0x1;

                inners_stats[idx+i].slot_use = residue-1;
                inners_stats[idx+i].bit_map = temp_bit_map;

                memcpy(&bt_inners[idx + i].keys,&nextlevelnonleafkeys[prev_inserted_inner],(residue-1)*sizeof(uint64_t));
                for (j=0; j<residue;j++){
                    bt_inners[idx + i].ptrs[j] = nextpointedblockid;
                    nextpointedblockid++;
                }

                j--;
                nextpointedblockid--;
	            for (;j<BLOCKSIZE_64BIT;j++){
		            bt_inners[idx + i].keys[j] =  ULONG_MAX;
                    bt_inners[idx + i].ptrs[j] = nextpointedblockid ;
                }
            }
    
            nextpointedblockid = idx;
        }

        //all the remaining inner levels, that we do not insert gaps(based on the height optimized mechanism)
        for (; level < bt_stats.height - 1 ; level ++){
            prev_inserted_inner = 0;
			
            uint32_t numptrs = NUMnextlevelnonleafkeys + 1; // include leftmost pointer which has no key
            NUMnextlevelnonleafkeys = 0;
					
            numcblocks = lut[level];
		    idx = 0;
            for (j = level + 1; j < bt_stats.height ; j ++){// calculate the position that each level will be saved in the array of inner nodes. bottom-up implementation
                idx += lut[j];// idx is used to write inner nodes to the appropriate position to the inner array from right to left(first left position is root)
            }

			//create all nodes in the same level, except the last one
            for (i=0; i<numcblocks-1; i++) {

                memcpy(&bt_inners[idx + i].keys, &nextlevelnonleafkeys[prev_inserted_inner], (BLOCKSIZE_64BIT - 1) * sizeof(uint64_t));
                bt_inners[idx + i].keys[BLOCKSIZE_64BIT - 1] = ULONG_MAX;
                
                inners_stats[idx+i].slot_use = BLOCKSIZE_64BIT-1;
                inners_stats[idx+i].bit_map = 0xffff;

                for (uint32_t t = 0; t < BLOCKSIZE_64BIT; t++)
                {
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid++;
                }
                prev_inserted_inner += BLOCKSIZE_64BIT -1;
                
                //update parent array for the next level
                nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = nextlevelnonleafkeys[prev_inserted_inner];
                prev_inserted_inner++;          
			}
            
            //create last node  
            uint32_t residue = numptrs%BLOCKSIZE_64BIT;
            uint16_t temp_bit_map = 0xffff;
            if(residue == 0) residue = BLOCKSIZE_64BIT;

            temp_bit_map = temp_bit_map << ( 16 - residue +1);
            temp_bit_map = temp_bit_map | 0x1;
            inners_stats[idx+i].slot_use = residue-1; 
            inners_stats[idx+i].bit_map = temp_bit_map;
            
            memcpy(&bt_inners[idx + i].keys,&nextlevelnonleafkeys[prev_inserted_inner],(residue-1)*sizeof(uint64_t));
            for (j=0; j<residue;j++){
                bt_inners[idx + i].ptrs[j] = nextpointedblockid;
                nextpointedblockid++;
            }

            j--;
            nextpointedblockid--;
            for (;j<BLOCKSIZE_64BIT;j++){
                bt_inners[idx + i].keys[j] =  ULONG_MAX;
                bt_inners[idx + i].ptrs[j] = nextpointedblockid ;
            }
    
            nextpointedblockid = idx;
        }
            
            
        //build root
        if (bt_stats.num_inners > 1){
            nextpointedblockid = 1;
        }
        else {
            nextpointedblockid = 0;
        }

        uint16_t temp_bit_map = 0xffff;
        temp_bit_map = temp_bit_map << (BLOCKSIZE_64BIT - NUMnextlevelnonleafkeys);
        temp_bit_map = temp_bit_map | 0x1;

        inners_stats[0].slot_use = NUMnextlevelnonleafkeys;
        inners_stats[0].bit_map = temp_bit_map;

        memcpy(&bt_inners[0].keys,nextlevelnonleafkeys,NUMnextlevelnonleafkeys*sizeof(uint64_t));
        
        for (j=0; j<=NUMnextlevelnonleafkeys;j++){
            bt_inners[0].ptrs[j] = nextpointedblockid;
            nextpointedblockid ++ ;
        }

        nextpointedblockid--;
        for (j = NUMnextlevelnonleafkeys;j<BLOCKSIZE_64BIT;j++){
            bt_inners[0].keys[j] = ULONG_MAX;
            bt_inners[0].ptrs[j] = nextpointedblockid;
        }        
    }
}


//search function, that traverses the bstree to find the skey
//same code with original bstree, but we have variable length in leaves
bool search_compact(uint64_t key){
    uint32_t i, pos, offset = root_offset;
    uint64_t skey = key;
    uint64_t keyInLeaf;
    
    for (i = bt_stats.height; i > 0; i--){
        __builtin_prefetch(bt_inners[offset].ptrs); //prefetch the pointers to the next node 
        pos = successorLinG_compact(bt_inners[offset].keys,skey);// search in inner nodes
        offset = bt_inners[offset].ptrs[pos];
    }


    //search in leaves
    uint8_t leaf_case = comp_leaves_stats[offset].leaf_case;
    pos = successorLinGE_compact(bt_comp_leaves[offset].keys, comp_leaves_stats[offset].reference_key, leaf_case ,&skey);

    switch (leaf_case){
        case 0: // 16-bit differences        
            keyInLeaf = (uint64_t) ((bt_comp_leaves[offset].keys[pos/2] >> (16 * (pos % 2))) & 0x0000ffff);
            break;

        case 1: // 32-bit differences
            keyInLeaf = (uint64_t) bt_comp_leaves[offset].keys[pos];
            break;
        
        case 2: //64-bit keys
            keyInLeaf = (uint64_t) bt_comp_leaves[offset].keys[2 * pos + 1];
            keyInLeaf = (keyInLeaf << 32) | bt_comp_leaves[offset].keys[2 * pos];
            break;
    }

    if (keyInLeaf==skey){
        return 1;
    }
    else{
        return 0;
    }
}


//insert function in the compact bstree
//General idea is that for a gap, at first we are looking at the right of the position and then at the right
//if a leaf node, does not have space, we split the node, and we insert the gaps proactively.
//Same functionality with the inners.
//same code with original bstree, but we have variable length in leavess
void insert_key_compact(uint64_t ikey){
    
    uint32_t i, pos, offset = root_offset;
    uint64_t skey = ikey;
    uint64_t key_for_next_level = 0;
    uint32_t ptr_for_next_level = 0;    
    uint64_t keyInLeaf;

    vector < pair <uint32_t, uint16_t>> tree_path;
    tree_path.resize(bt_stats.height);

    uint32_t offsets_path[bt_stats.height];


    for (i = bt_stats.height; i > 0; i--){ 
        __builtin_prefetch(bt_inners[offset].ptrs);
		pos = successorLinG_compact(bt_inners[offset].keys,skey);
        tree_path[i-1] = make_pair(offset, (uint16_t) pos);
        offsets_path[i-1] = offset;
        offset = bt_inners[offset].ptrs[pos];
    }


    uint8_t leaf_case = comp_leaves_stats[offset].leaf_case;

    pos = successorLinGE_compact(bt_comp_leaves[offset].keys, comp_leaves_stats[offset].reference_key, leaf_case ,&skey);

    //leaf nodes
    uint8_t slot_use = comp_leaves_stats[offset].slot_use;
 

    switch (leaf_case){
        case 0: //16 bit differences
            keyInLeaf = (uint64_t) ((bt_comp_leaves[offset].keys[pos/2] >> (16 * (pos % 2))) & 0x0000ffff);
                
            if (keyInLeaf == skey)
                return;
        

            if (slot_use < BLOCKSIZE_16BIT){   
                
                if (slot_use != 0){
                    uint64_t bit_map_16bit = comp_leaves_stats[offset].bit_map;
                    bit_map_16bit = ~bit_map_16bit;

                    uint16_t temp_array_16bit[BLOCKSIZE_16BIT];
                    
                    memcpy(&temp_array_16bit, &bt_comp_leaves[offset].keys, BLOCKSIZE_16BIT * sizeof(uint16_t));

                    uint32_t shift_pos_16bit;

                    if ((pos == BLOCKSIZE_16BIT - 1)){
                        shift_pos_16bit = _tzcnt_u64(bit_map_16bit);
                
                        if (shift_pos_16bit > 0){

                            uint64_t bit_flip_16bit = 0x0000000000000001;
                            uint32_t start_copy = pos - shift_pos_16bit;
                            uint32_t keysForCopy = shift_pos_16bit;

                            if (skey < keyInLeaf){
                                keysForCopy--;
                                pos--;
                            }

                            memcpy(&temp_array_16bit[start_copy], &temp_array_16bit[start_copy + 1], keysForCopy * sizeof(uint16_t));
                            temp_array_16bit[pos] = (uint16_t) skey;

                            bit_map_16bit = bit_map_16bit ^ (bit_flip_16bit << shift_pos_16bit);
                        }
                        else{
                            // printf("But the last entry is empty\n");
                            temp_array_16bit[BLOCKSIZE_16BIT - 1] = (uint16_t) skey;
                            bit_map_16bit = bit_map_16bit ^ 0x0000000000000001;
                        }
                    }
                    else{

                        if (temp_array_16bit[pos] == temp_array_16bit[pos + 1]){
                            temp_array_16bit[pos] = (uint16_t) skey;
                            bit_map_16bit = bit_map_16bit ^ (0x8000000000000000 >> pos);                    //00010001000100010 00100010001000100010001000100010001000100010001

                        }
                        else{
                            uint64_t bit_flip_16bit = 0x8000000000000000;
                            uint64_t temp_bit_map = bit_map_16bit << pos;
                            shift_pos_16bit = _lzcnt_u64(temp_bit_map);

                            if (shift_pos_16bit == 64){                                             // 0000000000000000000000000000000000000000000000000010001000100010 
                                temp_bit_map = bit_map_16bit >> (BLOCKSIZE_16BIT - pos - 1); 
                                shift_pos_16bit =_tzcnt_u64(temp_bit_map);

                                uint32_t start_copy = pos - shift_pos_16bit;
                                
                                memcpy(&temp_array_16bit[start_copy], &temp_array_16bit[start_copy + 1], (shift_pos_16bit-1) * sizeof(uint16_t));
                                bit_map_16bit = bit_map_16bit ^ (bit_flip_16bit >> start_copy);
                                temp_array_16bit[pos-1] = (uint16_t) skey;
                            }
                            else{    
                                uint32_t diff = shift_pos_16bit;
                                memcpy(&temp_array_16bit[pos + 1], &temp_array_16bit[pos], diff * sizeof(uint16_t));
                                
                                bit_map_16bit = bit_map_16bit ^ (bit_flip_16bit >> (pos + diff));
                                temp_array_16bit[pos] = (uint16_t) skey;
                            }
                        }
                    }

                    memcpy(&bt_comp_leaves[offset].keys, &temp_array_16bit, BLOCKSIZE_16BIT * sizeof(uint16_t));

                    comp_leaves_stats[offset].slot_use++;
                    comp_leaves_stats[offset].bit_map = ~bit_map_16bit;
                }
                else{
                    bt_comp_leaves[offset].keys[0] = 0xffff0000;
                    
                    comp_leaves_stats[offset].slot_use++;
                    comp_leaves_stats[offset].bit_map = 0x8000000000000000;
                    comp_leaves_stats[offset].reference_key = ikey;
                }

                return;
            }
            else{    // split is needed   
                
                if (bt_stats.num_leaves == bt_stats.max_num_leaves){
                    printf("I reached at the reallocation point in Leaf nodes!!!!\n\n");
                    reallocate_leaves_compact();
                }

                uint16_t total_keys[BLOCKSIZE_16BIT + 1];
                uint16_t temp_array_16bit[BLOCKSIZE_16BIT];

                memcpy(&total_keys, &bt_comp_leaves[offset].keys, BLOCKSIZE_16BIT * sizeof(uint16_t));
                
                if ((uint16_t) skey > total_keys[BLOCKSIZE_16BIT - 1]){
                    total_keys[pos + 1] = (uint16_t) skey;
                }
                else{
                    uint32_t keysForCopy = BLOCKSIZE_16BIT - pos;
                    memcpy(&total_keys[pos + 1], &total_keys[pos], keysForCopy * sizeof(uint16_t));
                    total_keys[pos] = (uint16_t) skey;
                }
                
            
                /* Create the original node with the new diffs of keys */
                temp_array_16bit[0] = total_keys[0];

                uint32_t j = 1;
                for (uint32_t i = 1; i < BLOCKSIZE_16BIT - 1; i+=2, j++){
                    temp_array_16bit[i] = total_keys[j];
                    temp_array_16bit[i + 1] = total_keys[j];
                }

                temp_array_16bit[BLOCKSIZE_16BIT - 1] = total_keys[j];

                memcpy(&bt_comp_leaves[offset].keys, &temp_array_16bit, BLOCKSIZE_16BIT * sizeof(uint16_t));
                
                comp_leaves_stats[offset].slot_use = 33;
                comp_leaves_stats[offset].bit_map = 0xaaaaaaaaaaaaaaab;

                /* Create the new node based on the differences of the remaining keys according the first key of this node */
                key_for_next_level = total_keys[++j] + comp_leaves_stats[offset].reference_key;
                ptr_for_next_level = bt_stats.num_leaves;

                j++;

                temp_array_16bit[0] = 0;

                for (uint32_t i = 1; i < BLOCKSIZE_16BIT - 1; i += 2, j++){
                    uint64_t reconstructedKey = total_keys[j] + comp_leaves_stats[offset].reference_key;
                    uint16_t insert_diff = (uint16_t) (reconstructedKey - key_for_next_level);

                    temp_array_16bit[i] = insert_diff;
                    temp_array_16bit[i + 1] = insert_diff;
                }

                temp_array_16bit[BLOCKSIZE_16BIT - 1] = USHRT_MAX;

                uint32_t new_offset = bt_stats.num_leaves;

                memcpy(&bt_comp_leaves[new_offset].keys, &temp_array_16bit, BLOCKSIZE_16BIT * sizeof(uint16_t));

                comp_leaves_stats[new_offset].leaf_case = 0;
                comp_leaves_stats[new_offset].slot_use = 32;
                comp_leaves_stats[new_offset].next_leaf = comp_leaves_stats[offset].next_leaf;
                comp_leaves_stats[new_offset].bit_map = 0xaaaaaaaaaaaaaaaa;
                comp_leaves_stats[new_offset].reference_key = key_for_next_level;

                comp_leaves_stats[offset].next_leaf = new_offset;
                
                bt_stats.num_leaves++;
            }
            break;
        
        case 1: // 32-bit differences

            keyInLeaf = (uint64_t) bt_comp_leaves[offset].keys[pos];
                
            if (keyInLeaf == skey)
                return;    
            
            
            if (slot_use < BLOCKSIZE_32BIT){

                if (slot_use != 0){
                    uint32_t bit_map_32bit = (uint32_t) comp_leaves_stats[offset].bit_map;
                    bit_map_32bit = ~bit_map_32bit;

                    uint32_t shift_pos_32bit;

                    if (pos == BLOCKSIZE_32BIT - 1){
                        shift_pos_32bit = _tzcnt_u32(bit_map_32bit);

                        if (shift_pos_32bit > 0){
                            uint32_t bit_flip_32bit = 0x00000001;
                            uint32_t start_copy = pos - shift_pos_32bit;
                            uint32_t keysForCopy = shift_pos_32bit;

                            if (skey < keyInLeaf){
                                keysForCopy--;
                                pos--;
                            }

                            memcpy(&bt_comp_leaves[offset].keys[start_copy], &bt_comp_leaves[offset].keys[start_copy + 1], keysForCopy * sizeof(uint32_t));

                            bt_comp_leaves[offset].keys[pos] = (uint32_t) skey;
                            bit_map_32bit = bit_map_32bit ^ (bit_flip_32bit << shift_pos_32bit);
                        }
                        else{
                            bt_comp_leaves[offset].keys[BLOCKSIZE_32BIT - 1] = (uint32_t) skey;
                            bit_map_32bit = bit_map_32bit ^ 0x00000001;
                        }
                    }
                    else{
                        if (bt_comp_leaves[offset].keys[pos] == bt_comp_leaves[offset].keys[pos+1]){
                            bt_comp_leaves[offset].keys[pos] = (uint32_t) skey;
                            bit_map_32bit = bit_map_32bit ^ (0x80000000 >> pos);                  

                        }
                        else{
                            uint32_t bit_flip_32bit = 0x80000000;
                            uint32_t temp_bit_map = bit_map_32bit << pos;
                            shift_pos_32bit = _lzcnt_u32(temp_bit_map);

                            if (shift_pos_32bit == 32){                                              
                                temp_bit_map = bit_map_32bit >> (BLOCKSIZE_32BIT - pos - 1); 
                                shift_pos_32bit =_tzcnt_u32(temp_bit_map);

                                uint32_t start_copy = pos - shift_pos_32bit;
                                
                                memcpy(&bt_comp_leaves[offset].keys[start_copy], &bt_comp_leaves[offset].keys[start_copy + 1], (shift_pos_32bit-1) * sizeof(uint32_t));
                                
                                
                                bit_map_32bit = bit_map_32bit ^ (bit_flip_32bit >> start_copy);
                                bt_comp_leaves[offset].keys[pos-1] = (uint32_t) skey;
                            }
                            else{    
                                uint32_t diff = shift_pos_32bit;

                                memcpy(&bt_comp_leaves[offset].keys[pos + 1], &bt_comp_leaves[offset].keys[pos], diff * sizeof(uint32_t));
                                
                                bit_map_32bit = bit_map_32bit ^ (bit_flip_32bit >> (pos + diff));
                                bt_comp_leaves[offset].keys[pos] = (uint32_t) skey;
                            }
                        }
                    }

                    comp_leaves_stats[offset].slot_use++;
                    comp_leaves_stats[offset].bit_map = ~bit_map_32bit;
                }
                else{
                    bt_comp_leaves[offset].keys[0] = 0;
                    
                    comp_leaves_stats[offset].slot_use++;
                    comp_leaves_stats[offset].bit_map = 0x80000000;
                    comp_leaves_stats[offset].reference_key = ikey;

                }

                return;

            }
            else{    // split is needed.
                
                if (bt_stats.num_leaves == bt_stats.max_num_leaves){
                    printf("I reached at the reallocation point in Leaf nodes!!!!\n\n");
                    reallocate_leaves_compact();
                }
        
                uint32_t total_keys[BLOCKSIZE_32BIT+1];
            
                memcpy(&total_keys, &bt_comp_leaves[offset].keys, pos * sizeof(uint32_t));
                
                if (skey > bt_comp_leaves[offset].keys[BLOCKSIZE_32BIT - 1]){
                    total_keys[pos] = bt_comp_leaves[offset].keys[BLOCKSIZE_32BIT - 1];
                    total_keys[pos + 1] = (uint32_t) skey;
                }
                else{
                    total_keys[pos] = (uint32_t) skey;
                    memcpy(&total_keys[pos + 1], &bt_comp_leaves[offset].keys[pos], (BLOCKSIZE_32BIT - pos) * sizeof(uint32_t));

                }

                /*  Create the original node with the appropriate diffs */
                bt_comp_leaves[offset].keys[0] = total_keys[0];

                uint32_t j = 1;
                for (uint32_t i = 1; i < BLOCKSIZE_32BIT - 1;  i+= 2, j++){
                    bt_comp_leaves[offset].keys[i] = total_keys[j];
                    bt_comp_leaves[offset].keys[i + 1] = total_keys[j];
                }

                bt_comp_leaves[offset].keys[BLOCKSIZE_32BIT - 1] = total_keys[j];

                comp_leaves_stats[offset].slot_use = 17;
                comp_leaves_stats[offset].bit_map = 0xaaaaaaab;

                /* Create the new node based on the differences of the remaining keys according the first key of this node */
                key_for_next_level = total_keys[++j] + comp_leaves_stats[offset].reference_key;
                ptr_for_next_level = bt_stats.num_leaves;

                j++;

                uint64_t parent = bt_inners[tree_path[0].first].keys[tree_path[0].second];
                uint64_t diff = parent - key_for_next_level;
                uint8_t leading_zeros = _lzcnt_u64(diff);

                uint32_t new_offset = bt_stats.num_leaves;

                if (leading_zeros >= 48){   
                    uint16_t temp_array_16bit[BLOCKSIZE_16BIT];
                    
                    temp_array_16bit[0] = 0;

                    for (uint32_t i = 1; i < BLOCKSIZE_16BIT - 3; i += 4, j++){
                        uint64_t reconstructedKey = total_keys[j] + comp_leaves_stats[offset].reference_key;
                        uint16_t inserted_diff = (uint16_t) (reconstructedKey - key_for_next_level);

                        for (uint32_t k = 0; k <= 3; k++){
                            temp_array_16bit[i + k] = inserted_diff;
                        }
                    }

                    temp_array_16bit[BLOCKSIZE_16BIT - 3] = USHRT_MAX;
                    temp_array_16bit[BLOCKSIZE_16BIT - 1] = temp_array_16bit[BLOCKSIZE_16BIT - 2] = temp_array_16bit[BLOCKSIZE_16BIT - 3];                

                    memcpy(&bt_comp_leaves[new_offset].keys, &temp_array_16bit, BLOCKSIZE_16BIT * sizeof(uint16_t));

                    comp_leaves_stats[new_offset].leaf_case = 0;
                    comp_leaves_stats[new_offset].slot_use = 16;
                    comp_leaves_stats[new_offset].next_leaf = comp_leaves_stats[offset].next_leaf;
                    comp_leaves_stats[new_offset].bit_map = 0x8888888888888888;
                
                }
                else {
                    bt_comp_leaves[new_offset].keys[0] = 0;

                    for (uint32_t i = 1; i < BLOCKSIZE_32BIT - 1; i+=2, j++){
                        uint64_t reconstructedKey = total_keys[j] + comp_leaves_stats[offset].reference_key;
                        uint32_t insert_diff = (uint32_t) (reconstructedKey - key_for_next_level);

                        bt_comp_leaves[new_offset].keys[i] = insert_diff;
                        bt_comp_leaves[new_offset].keys[i + 1] = insert_diff;
                    }

                    bt_comp_leaves[new_offset].keys[BLOCKSIZE_32BIT - 1] = UINT_MAX;
                    
                    comp_leaves_stats[new_offset].leaf_case = 1;
                    comp_leaves_stats[new_offset].slot_use = 16;
                    comp_leaves_stats[new_offset].next_leaf = comp_leaves_stats[offset].next_leaf;
                    comp_leaves_stats[new_offset].bit_map = 0xaaaaaaaa;
                }

                comp_leaves_stats[new_offset].reference_key = key_for_next_level;
                comp_leaves_stats[offset].next_leaf = new_offset;
                bt_stats.num_leaves++;
            }
            
            break;
        
        case 2://64-bit keys

            keyInLeaf = (uint64_t) bt_comp_leaves[offset].keys[2 * pos + 1];
            keyInLeaf = (keyInLeaf << 32) | bt_comp_leaves[offset].keys[2 * pos];

            if (keyInLeaf == skey)
                return;
            
            if (slot_use < BLOCKSIZE_64BIT){
                
                if (slot_use != 0){
                    uint16_t bit_map_64bit = (uint16_t) comp_leaves_stats[offset].bit_map;
                    bit_map_64bit = ~bit_map_64bit;
                    
                    uint64_t temp_array_64bit[BLOCKSIZE_64BIT];
                    memcpy(&temp_array_64bit, &bt_comp_leaves[offset].keys, BLOCKSIZE_64BIT * sizeof(uint64_t));
        
                    uint32_t shift_pos_64bit;

                    if (pos == 0){

                        shift_pos_64bit = _lzcnt_u32((uint32_t)bit_map_64bit);
                        
                        if (shift_pos_64bit > 16){
                            uint16_t bit_flip_64bit = 0x8000;
                            uint32_t diff = shift_pos_64bit - 16;

                            memcpy(&temp_array_64bit[pos + 1], &temp_array_64bit[pos], diff * sizeof(uint64_t));                    
                            bit_map_64bit = bit_map_64bit ^ (bit_flip_64bit >> diff);
                        }
                        else{
                            bit_map_64bit = bit_map_64bit ^ 0x8000;
                        }
                        
                        temp_array_64bit[pos] = skey;
                    
                    }
                    else if (pos == BLOCKSIZE_64BIT - 1){

                        shift_pos_64bit = _tzcnt_u16(bit_map_64bit);
                    
                        if (shift_pos_64bit > 0){
                            uint16_t bit_flip_64bit = 0x0001;
                            uint32_t start_copy = pos - shift_pos_64bit;
                            uint32_t keysForCopy = shift_pos_64bit;

                            if (skey < keyInLeaf){
                                keysForCopy--;
                                pos--;
                            }

                            memcpy(&temp_array_64bit[start_copy], &temp_array_64bit[start_copy + 1], keysForCopy * sizeof(uint64_t));

                            bit_map_64bit = bit_map_64bit ^ (bit_flip_64bit << shift_pos_64bit);
                            temp_array_64bit[pos] = skey;
                        }
                        else{
                            bit_map_64bit = bit_map_64bit ^ 0x0001;
                            temp_array_64bit[pos] = skey;
                        }
                    }
                    else{

                        if (temp_array_64bit[pos] == temp_array_64bit[pos + 1]){
                            temp_array_64bit[pos] = skey;
                            bit_map_64bit = bit_map_64bit ^ (0x8000 >> pos);
                        }
                        else{
                            
                            uint16_t bit_flip_64bit = 0x8000;
                            uint32_t temp_bit_map = bit_map_64bit << (pos + 16);
                            shift_pos_64bit = _lzcnt_u32(temp_bit_map);

                            if (shift_pos_64bit == 32){
                                temp_bit_map = bit_map_64bit >> (16 - pos - 1); 
                                shift_pos_64bit = _tzcnt_u32(temp_bit_map);
                                
                                uint32_t start_copy = pos - shift_pos_64bit;
                                
                                memcpy(&temp_array_64bit[start_copy], &temp_array_64bit[start_copy + 1], (shift_pos_64bit-1) * sizeof(uint64_t));

                                bit_map_64bit = bit_map_64bit ^ (bit_flip_64bit >> start_copy);
                                temp_array_64bit[pos-1] = skey;
                            }
                            else{ 

                                uint32_t diff = shift_pos_64bit;
                                memcpy(&temp_array_64bit[pos + 1], &temp_array_64bit[pos], diff * sizeof(uint64_t));
                                
                                bit_map_64bit = bit_map_64bit ^ (bit_flip_64bit >> (pos + diff));
                                temp_array_64bit[pos] = skey;
                            }
                        }
                    }

                    memcpy(&bt_comp_leaves[offset].keys, &temp_array_64bit, BLOCKSIZE_64BIT * sizeof(uint64_t));

                    comp_leaves_stats[offset].slot_use++;
                    comp_leaves_stats[offset].bit_map = ~bit_map_64bit;
                }
                else{
                    bt_comp_leaves[offset].keys[0] = (uint32_t) ((skey & 0xffffffff00000000) >> 32);
                    bt_comp_leaves[offset].keys[1] = (uint32_t) (skey & 0x00000000ffffffff);
                    
                    comp_leaves_stats[offset].slot_use++;
                    comp_leaves_stats[offset].bit_map = 0x8000;
                    comp_leaves_stats[offset].reference_key = ikey;
                }

                return;
            }
            else{    // split is needed
                
                if (bt_stats.num_leaves == bt_stats.max_num_leaves){
                    printf("I reached at the reallocation point in Leaf nodes!!!!\n\n");
                    reallocate_leaves_compact();
                }
        
                uint64_t total_keys[BLOCKSIZE_64BIT+1];

                /* Create the left node with 9 keys and 7 gaps */
                uint64_t temp_array_64bit[BLOCKSIZE_64BIT];

                //create temp array with BLOCKSIZE + 1
                memcpy(&total_keys, &bt_comp_leaves[offset].keys, BLOCKSIZE_64BIT * sizeof(uint64_t));
                
                if (ikey > total_keys[BLOCKSIZE_64BIT -1]){ 
                    total_keys[pos + 1] = ikey;
                }
                else{
                    uint32_t keysForCopy = BLOCKSIZE_64BIT - pos;
                    memcpy(&total_keys[pos + 1], &total_keys[pos], keysForCopy * sizeof(uint64_t));
                    total_keys[pos] = ikey;
                }
                
                temp_array_64bit[0] = total_keys[0];

                uint32_t j = 1;
                for (uint32_t i = 1; i < BLOCKSIZE_64BIT - 1;  i+= 2, j++){
                    temp_array_64bit[i] = total_keys[j];
                    temp_array_64bit[i + 1] = total_keys[j];
                }

                temp_array_64bit[BLOCKSIZE_64BIT - 1] = total_keys[j];

                memcpy(&bt_comp_leaves[offset].keys, &temp_array_64bit, BLOCKSIZE_64BIT * sizeof(uint64_t));

                comp_leaves_stats[offset].slot_use = 9;
                comp_leaves_stats[offset].bit_map = 0xaaab;

                /* Create the new node with the remaining keys or the relative diffs if it's possible */

                key_for_next_level = total_keys[++j];
                ptr_for_next_level = bt_stats.num_leaves;

                j++;

                uint64_t parent = bt_inners[tree_path[0].first].keys[tree_path[0].second];
                uint64_t diff = parent - key_for_next_level;
                uint8_t leading_zeros = _lzcnt_u64(diff);

                uint32_t new_offset = bt_stats.num_leaves;

                if (leading_zeros >= 48){    // this node can be expressed as diffs with 16bit info
                    uint16_t temp_array_16bit[BLOCKSIZE_16BIT];

                    temp_array_16bit[0] = 0;
                    for (uint32_t i = 1; i < BLOCKSIZE_16BIT - 1; i+=9, j++){
                        uint16_t inserted_diff = (uint16_t) (total_keys[j] - key_for_next_level);
                        for (uint32_t k = 0; k <= 8; k++){
                            temp_array_16bit[i + k] = inserted_diff;
                        }
                    }
                    
                    memcpy(&bt_comp_leaves[new_offset].keys, &temp_array_16bit, BLOCKSIZE_16BIT * sizeof(uint16_t));

                    comp_leaves_stats[new_offset].leaf_case = 0;
                    comp_leaves_stats[new_offset].slot_use = 8;
                    comp_leaves_stats[new_offset].next_leaf = comp_leaves_stats[offset].next_leaf;
                    comp_leaves_stats[new_offset].bit_map = 0x8040201008040201;

                }
                else if (leading_zeros >= 32){
                    bt_comp_leaves[new_offset].keys[0] = 0;

                    for (uint32_t i = 1; i < BLOCKSIZE_32BIT -3; i+= 4, j++){
                        uint32_t inserted_diff = (uint32_t) (total_keys[j] - key_for_next_level);
                        for (uint32_t k = 0; k <= 3; k++){
                            bt_comp_leaves[new_offset].keys[i + k] = inserted_diff;
                        }
                    }

                    bt_comp_leaves[new_offset].keys[BLOCKSIZE_32BIT - 3] = UINT_MAX;
                    bt_comp_leaves[new_offset].keys[BLOCKSIZE_32BIT - 1] = bt_comp_leaves[new_offset].keys[BLOCKSIZE_32BIT - 2] = bt_comp_leaves[new_offset].keys[BLOCKSIZE_32BIT - 3];

                    comp_leaves_stats[new_offset].leaf_case = 1;
                    comp_leaves_stats[new_offset].slot_use = 8;
                    comp_leaves_stats[new_offset].next_leaf = comp_leaves_stats[offset].next_leaf;
                    comp_leaves_stats[new_offset].bit_map = 0x88888888;
                }
                else{
                    temp_array_64bit[0] = key_for_next_level;
                    for (uint32_t i = 1; i < BLOCKSIZE_64BIT - 1; i += 2, j++){
                        temp_array_64bit[i] = total_keys[j];
                        temp_array_64bit[i + 1] = total_keys[j];
                    }

                    temp_array_64bit[BLOCKSIZE_64BIT - 1] = ULONG_MAX;

                    memcpy(&bt_comp_leaves[new_offset].keys, &temp_array_64bit, BLOCKSIZE_64BIT * sizeof(uint64_t));

                    comp_leaves_stats[new_offset].leaf_case = 2;
                    comp_leaves_stats[new_offset].slot_use = 8;
                    comp_leaves_stats[new_offset].next_leaf = comp_leaves_stats[offset].next_leaf;
                    comp_leaves_stats[new_offset].bit_map = 0xaaaa;

                }

                comp_leaves_stats[new_offset].reference_key = key_for_next_level;
                comp_leaves_stats[offset].reference_key = total_keys[0];
                comp_leaves_stats[offset].next_leaf = new_offset;
                bt_stats.num_leaves++;
            }

            break; 
    }


    //insert in the inners above
    for (uint32_t j = 0 ; j < bt_stats.height; j++){
        
        offset = tree_path[j].first;
        pos = tree_path[j].second;

        slot_use = inners_stats[offset].slot_use;

        if (slot_use < BLOCKSIZE_64BIT -1){//we have empty slot
            uint32_t shift_pos = 0;
            uint16_t bit_map = inners_stats[offset].bit_map;
            bit_map = ~bit_map;
        
            if (bt_inners[offset].keys[pos] == bt_inners[offset].keys[pos + 1]){
                //update key
                bt_inners[offset].keys[pos] = key_for_next_level;

                bit_map = bit_map ^ (0x8000 >> pos);


                if (bt_inners[offset].keys[pos + 1] != ULONG_MAX){
                    bt_inners[offset].ptrs[pos + 1] = ptr_for_next_level;
                }
                else{
                    for (uint32_t p = pos + 1; p < BLOCKSIZE_64BIT; p++){
                        bt_inners[offset].ptrs[p] = ptr_for_next_level;
                    }
                }

            }
            else{
                uint32_t temp_bit_map = bit_map << (pos + 16);
                shift_pos = _lzcnt_u32(temp_bit_map);

                if (shift_pos == 32){
                    uint16_t bit_flip = 0x8000;

                    temp_bit_map = bit_map >> (16 - pos - 1); 
                    shift_pos =_tzcnt_u32(temp_bit_map);
                    
                    //update keys
                    uint32_t start_copy = pos - shift_pos;
                    memcpy(&bt_inners[offset].keys[start_copy], &bt_inners[offset].keys[start_copy + 1] , (shift_pos-1) * sizeof(uint64_t) );
                    bit_map = bit_map ^ (bit_flip >> start_copy);
                    bt_inners[offset].keys[pos-1] = key_for_next_level;

                    //update pointers
                    memcpy(&bt_inners[offset].ptrs[start_copy], &bt_inners[offset].ptrs[start_copy + 1] , shift_pos* sizeof(uint32_t) );
                    bt_inners[offset].ptrs[pos] = ptr_for_next_level;
                }
                else{

                    //update keys
                    uint16_t bit_flip = 0x8000;
                    uint32_t diff = shift_pos;

                    memcpy(&bt_inners[offset].keys[pos + 1], &bt_inners[offset].keys[pos] , diff * sizeof(uint64_t) );
                    bit_map = bit_map ^ (bit_flip >> (pos + diff));
                    bt_inners[offset].keys[pos] = key_for_next_level;
                    
                    memcpy(&bt_inners[offset].ptrs[pos + 1], &bt_inners[offset].ptrs[pos] , diff  * sizeof(uint32_t) );
                    bt_inners[offset].ptrs[pos+1] = ptr_for_next_level;
                }
            }

            inners_stats[offset].slot_use++;
            inners_stats[offset].bit_map = ~bit_map;
        
            break;

        }
        else{// no empty slot, split

            if (bt_stats.num_inners == bt_stats.max_num_inners){
                printf("reallocation at inner nodes is needed!!!\n");
                reallocate_inners_compact();
            }
            
            uint32_t mid_in_node = BLOCKSIZE_64BIT/2;

            //insert the new key in the node and then make the split to have better load balancing
            if (pos > mid_in_node){ // insert in the right inner node
                
                //bring the pos position to the new node starting from zero
                pos = pos - mid_in_node - 1;
                
                //put keys and pointers for the first section (before pos) to the new node
                uint32_t k = mid_in_node + 1;
                for (i = 0 ; i < 2*pos ; i += 2){
                    bt_inners[bt_stats.num_inners].keys[i] = bt_inners[offset].keys[k];
                    bt_inners[bt_stats.num_inners].keys[i+1] = bt_inners[offset].keys[k];

                    bt_inners[bt_stats.num_inners].ptrs[i] = bt_inners[offset].ptrs[k];
                    bt_inners[bt_stats.num_inners].ptrs[i + 1] = bt_inners[offset].ptrs[k]; 
                    k++;
                }

                //put pos key and ptr to the new node
                bt_inners[bt_stats.num_inners].keys[i] = key_for_next_level;
                bt_inners[bt_stats.num_inners].keys[i+1] = key_for_next_level;
                
                bt_inners[bt_stats.num_inners].ptrs[i] = bt_inners[offset].ptrs[k];
                bt_inners[bt_stats.num_inners].ptrs[i+1]= bt_inners[offset].ptrs[k];

                //update the new ptr in the new node
                uint32_t update_ptr = ptr_for_next_level;

                //put keys and pointers for the second section (after pos to the end) to the new node
                for (i = i + 2; i < BLOCKSIZE_64BIT - 2 ; i += 2){
                    bt_inners[bt_stats.num_inners].keys[i] = bt_inners[offset].keys[k];
                    bt_inners[bt_stats.num_inners].keys[i+1] = bt_inners[offset].keys[k];

                    bt_inners[bt_stats.num_inners].ptrs[i] = update_ptr;
                    bt_inners[bt_stats.num_inners].ptrs[i+1] = update_ptr;

                    k++;

                    update_ptr = bt_inners[offset].ptrs[k]; 
                }
            
                //put in the last two positions the keys and the ptrs
                bt_inners[bt_stats.num_inners].keys[BLOCKSIZE_64BIT-2] = ULONG_MAX;
                bt_inners[bt_stats.num_inners].keys[BLOCKSIZE_64BIT-1] = ULONG_MAX;
                
                bt_inners[bt_stats.num_inners].ptrs[BLOCKSIZE_64BIT-2] = update_ptr;
                bt_inners[bt_stats.num_inners].ptrs[BLOCKSIZE_64BIT-1] = update_ptr;
                
                //new key for the upper level;
                key_for_next_level = bt_inners[offset].keys[mid_in_node];

                //update last pointer in the previous node
                bt_inners[offset].ptrs[BLOCKSIZE_64BIT-1] = bt_inners[offset].ptrs[mid_in_node];

                //update keys and ptrs for the previous node
                k = mid_in_node - 1;
                for ( i = BLOCKSIZE_64BIT -2 ; i > 0 ; i -= 2 ){
                    bt_inners[offset].keys[i] = bt_inners[offset].keys[k];
                    bt_inners[offset].keys[i-1] = bt_inners[offset].keys[k];

                    bt_inners[offset].ptrs[i] = bt_inners[offset].ptrs[k];
                    bt_inners[offset].ptrs[i-1] = bt_inners[offset].ptrs[k];

                    k--;
                }

                //update new ptr
                ptr_for_next_level = bt_stats.num_inners;

            }
            else{
                if (pos == mid_in_node){ // situation that the same key inserted in the upper level

                    uint32_t update_ptr = ptr_for_next_level;

                    //create new node and put the keys and ptrs
                    uint32_t k = mid_in_node;
                    for (i = 0 ; i < BLOCKSIZE_64BIT - 2 ; i += 2 ){
                        bt_inners[bt_stats.num_inners].keys[i] = bt_inners[offset].keys[k];
                        bt_inners[bt_stats.num_inners].keys[i + 1] = bt_inners[offset].keys[k];
                        
                        bt_inners[bt_stats.num_inners].ptrs[i] = update_ptr;
                        bt_inners[bt_stats.num_inners].ptrs[i + 1] = update_ptr;
                        
                        k++;

                        update_ptr =  bt_inners[offset].ptrs[k];                        
                    }

                    //put last 2 keys in the new node
                    bt_inners[bt_stats.num_inners].keys[BLOCKSIZE_64BIT-2] = ULONG_MAX;
                    bt_inners[bt_stats.num_inners].keys[BLOCKSIZE_64BIT-1] = ULONG_MAX;

                    //put last 2 pointers in the new node
                    bt_inners[bt_stats.num_inners].ptrs[BLOCKSIZE_64BIT-2] = update_ptr;
                    bt_inners[bt_stats.num_inners].ptrs[BLOCKSIZE_64BIT-1] = update_ptr;


                    //update last pointer in the previous node
                    bt_inners[offset].ptrs[BLOCKSIZE_64BIT-1] = bt_inners[offset].ptrs[mid_in_node];

                    //update keys and ptrs for the previous node
                    k = mid_in_node - 1;
                    for ( i = BLOCKSIZE_64BIT -2 ; i > 0 ; i -= 2 ){
                        bt_inners[offset].keys[i] = bt_inners[offset].keys[k];
                        bt_inners[offset].keys[i-1] = bt_inners[offset].keys[k];

                        bt_inners[offset].ptrs[i] = bt_inners[offset].ptrs[k];
                        bt_inners[offset].ptrs[i-1] = bt_inners[offset].ptrs[k];

                        k--;
                    }

                    //update new ptr
                    ptr_for_next_level = bt_stats.num_inners;

                }
                else{//insert in the left node

                    //create new node and put the keys and ptrs
                    uint32_t k = mid_in_node;
                    for (i = 0 ; i < BLOCKSIZE_64BIT - 2 ; i += 2 ){
                        bt_inners[bt_stats.num_inners].keys[i] = bt_inners[offset].keys[k];
                        bt_inners[bt_stats.num_inners].keys[i + 1] = bt_inners[offset].keys[k];
                        
                        bt_inners[bt_stats.num_inners].ptrs[i] = bt_inners[offset].ptrs[k];
                        bt_inners[bt_stats.num_inners].ptrs[i + 1] = bt_inners[offset].ptrs[k];
                        
                        k++;                        
                    }

                    //put last 2 keys in the new node
                    bt_inners[bt_stats.num_inners].keys[BLOCKSIZE_64BIT-2] = ULONG_MAX;
                    bt_inners[bt_stats.num_inners].keys[BLOCKSIZE_64BIT-1] = ULONG_MAX;


                    //put last 2 pointers in the new node
                    bt_inners[bt_stats.num_inners].ptrs[BLOCKSIZE_64BIT-2] = bt_inners[offset].ptrs[k];
                    bt_inners[bt_stats.num_inners].ptrs[BLOCKSIZE_64BIT-1] = bt_inners[offset].ptrs[k];


                    //update key for next level
                    uint64_t inserted_in_pos = key_for_next_level;
                    key_for_next_level = bt_inners[offset].keys[mid_in_node-1];
                    
                    // if the inserted key is stored at the last entry of the previous node
                    if (pos == mid_in_node - 1){
                        bt_inners[offset].keys[BLOCKSIZE_64BIT - 1] = ULONG_MAX;
                        bt_inners[offset].ptrs[BLOCKSIZE_64BIT - 1] = ptr_for_next_level;

                        bt_inners[offset].keys[BLOCKSIZE_64BIT - 2] = inserted_in_pos;
                        bt_inners[offset].keys[BLOCKSIZE_64BIT - 3] = inserted_in_pos;
                        
                        bt_inners[offset].ptrs[BLOCKSIZE_64BIT - 2] = bt_inners[offset].ptrs[mid_in_node - 1];
                        bt_inners[offset].ptrs[BLOCKSIZE_64BIT - 3] = bt_inners[offset].ptrs[mid_in_node - 1];

                        k = mid_in_node - 2;

                        for (i = BLOCKSIZE_64BIT - 4; i > 0; i -= 2){
                            bt_inners[offset].keys[i] = bt_inners[offset].keys[k];
                            bt_inners[offset].keys[i-1] = bt_inners[offset].keys[k];

                            bt_inners[offset].ptrs[i] = bt_inners[offset].ptrs[k];
                            bt_inners[offset].ptrs[i-1] = bt_inners[offset].ptrs[k];

                            k--;
                        }
                    }
                    else {  // otherwise we must to store the key between the keys that already has the previous node
                        
                        bt_inners[offset].keys[BLOCKSIZE_64BIT - 1] = ULONG_MAX;
                        bt_inners[offset].ptrs[BLOCKSIZE_64BIT - 1] = bt_inners[offset].ptrs[mid_in_node - 1];

                        k = mid_in_node - 2;

                        for (i = BLOCKSIZE_64BIT - 2; i > 2 * (pos + 1); i-= 2){
                            bt_inners[offset].keys[i] = bt_inners[offset].keys[k];
                            bt_inners[offset].keys[i-1] = bt_inners[offset].keys[k];

                            bt_inners[offset].ptrs[i] = bt_inners[offset].ptrs[k];
                            bt_inners[offset].ptrs[i-1] = bt_inners[offset].ptrs[k];

                            k--;
                        }

                        bt_inners[offset].keys[i] = bt_inners[offset].keys[k];
                        bt_inners[offset].keys[i-1] = bt_inners[offset].keys[k];

                        bt_inners[offset].ptrs[i] = ptr_for_next_level;
                        bt_inners[offset].ptrs[i-1] = ptr_for_next_level;

                        uint64_t update_key = inserted_in_pos;

                        for (i = i -2; i > 0; i-= 2){

                            bt_inners[offset].keys[i] = update_key;
                            bt_inners[offset].keys[i-1] = update_key;

                            bt_inners[offset].ptrs[i] = bt_inners[offset].ptrs[k];
                            bt_inners[offset].ptrs[i-1] = bt_inners[offset].ptrs[k];

                            k--;

                            update_key = bt_inners[offset].keys[k];
                        }

                        bt_inners[offset].keys[i] = update_key;
                        bt_inners[offset].ptrs[i] = bt_inners[offset].ptrs[k];

                    }

                    //update ptr for next level
                    ptr_for_next_level = bt_stats.num_inners;

                }

            }
            //slotuse and bitmaps for previous node
            inners_stats[offset].slot_use = mid_in_node;
            inners_stats[offset].bit_map = 0xaaab;

            //slotuse and bitmaps for new node
            inners_stats[bt_stats.num_inners].slot_use = mid_in_node - 1;
            inners_stats[bt_stats.num_inners].bit_map = 0x5555;
            
            bt_stats.num_inners++;

            if (offset == root_offset){
                root_offset = bt_stats.num_inners;

                bt_inners[root_offset].keys[0] = key_for_next_level;
                bt_inners[root_offset].ptrs[0] = offset;

                uint32_t last_pointer = ptr_for_next_level;

                for (uint32_t l = 1; l < BLOCKSIZE_64BIT; l++){
                    bt_inners[root_offset].keys[l] = ULONG_MAX;
                    bt_inners[root_offset].ptrs[l] = last_pointer;
                }

                inners_stats[root_offset].slot_use = 1;
                inners_stats[root_offset].bit_map = 0x8001;

                bt_stats.num_inners++;
                bt_stats.height++;

                return;            
            }
        }
    }
}


//delete key from the compact bstree
//same code with original bstree, but we have variable length in leaves
void delete_key_compact(uint64_t dkey){
    uint32_t i, pos, offset = root_offset;
    uint64_t skey = dkey;
    uint64_t keyInLeaf;
    uint32_t valid_pos;
    uint8_t slot_use;
    uint64_t bit_map_16bit;
    uint32_t bit_map_32bit;
    uint16_t bit_map_64bit;


    vector < pair <uint32_t, uint16_t>> tree_path; //vector to store the path from the root to leaf. (pair (offset of the node, position in the node))
    tree_path.resize(bt_stats.height);

    //search to find the position of the key, that we try to delete. 
    //search in inners
    for (i = bt_stats.height; i > 0; i--){
        
        __builtin_prefetch(bt_inners[offset].ptrs);
        pos = successorLinG_compact(bt_inners[offset].keys,skey);
        tree_path[i-1] = make_pair(offset, (uint16_t) pos);
        offset = bt_inners[offset].ptrs[pos];
    }

    //search in leafs
    uint8_t leaf_case = comp_leaves_stats[offset].leaf_case;
    pos = successorLinGE_compact(bt_comp_leaves[offset].keys, comp_leaves_stats[offset].reference_key, leaf_case ,&skey);
    slot_use = comp_leaves_stats[offset].slot_use;
    
    switch (leaf_case){
        case 0: // 16-bit differences

            keyInLeaf = (uint64_t) ((bt_comp_leaves[offset].keys[pos/2] >> (16 * (pos % 2))) & 0x0000ffff);
            
            if (keyInLeaf != skey)
                return; 
            
            if (slot_use == 1){

                for ( i = 0; i < BLOCKSIZE_32BIT; i++){
                    bt_comp_leaves[offset].keys[i] = UINT_MAX;
                }
                comp_leaves_stats[offset].slot_use = 0;
                comp_leaves_stats[offset].bit_map = 0;

                return;
            } 
            
            bit_map_16bit = comp_leaves_stats[offset].bit_map;
            
            uint16_t temp_array_16bit[BLOCKSIZE_16BIT];
            memcpy(&temp_array_16bit, &bt_comp_leaves[offset].keys, BLOCKSIZE_16BIT * sizeof(uint16_t));

            if (pos != BLOCKSIZE_16BIT - 1)        {
                valid_pos = _lzcnt_u64(bit_map_16bit << pos);

                uint16_t reproducedKey = 0;

                if (pos + valid_pos == BLOCKSIZE_16BIT - 1){
                    reproducedKey = USHRT_MAX;
                }
                else{
                    reproducedKey = temp_array_16bit[pos + 1 + valid_pos];
                }

                for (i = 0; i <= valid_pos; i++){
                    temp_array_16bit[pos + i] = reproducedKey;
                }

                bit_map_16bit = bit_map_16bit ^ (0x8000000000000000 >> (pos + valid_pos));

            }
            else{
                bit_map_16bit = bit_map_16bit ^ 0x0000000000000001;
                valid_pos = _tzcnt_u64(bit_map_16bit);

                for (i = 0; i < valid_pos; i++){
                    temp_array_16bit[pos - i] = USHRT_MAX;
                }
            }

            memcpy(&bt_comp_leaves[offset].keys, &temp_array_16bit, BLOCKSIZE_16BIT * sizeof(uint16_t));
            
            comp_leaves_stats[offset].slot_use--;
            comp_leaves_stats[offset].bit_map = bit_map_16bit;

            break;
        
        case 1:// 32-bit differences
            
            keyInLeaf = (uint64_t) bt_comp_leaves[offset].keys[pos];
        
            if (keyInLeaf != skey)
                return;

            
            if (slot_use == 1){

                for (i = 0; i < BLOCKSIZE_32BIT; i++){
                    bt_comp_leaves[offset].keys[i] = UINT_MAX;
                }

                comp_leaves_stats[offset].slot_use = 0;
                comp_leaves_stats[offset].bit_map = 0;

                return;
            }

            bit_map_32bit = (uint32_t) comp_leaves_stats[offset].bit_map;

            if (pos != BLOCKSIZE_32BIT - 1){
                valid_pos = _lzcnt_u32(bit_map_32bit << pos);

                uint32_t reproducedKey = 0;

                if (pos + valid_pos == BLOCKSIZE_32BIT - 1){
                    reproducedKey = UINT_MAX;
                }
                else{
                    reproducedKey = bt_comp_leaves[offset].keys[pos + 1 + valid_pos];
                }

                for (i = 0; i <= valid_pos; i++){
                    bt_comp_leaves[offset].keys[pos + i] = reproducedKey;
                }

                bit_map_32bit = bit_map_32bit ^ (0x80000000 >> (pos + valid_pos));

            }
            else{
                bit_map_32bit = bit_map_32bit ^ 0x00000001;
                valid_pos = _tzcnt_u32(bit_map_32bit);

                for (i = 0; i < valid_pos; i++){
                    bt_comp_leaves[offset].keys[pos - i] = UINT_MAX;
                }
            }
            
            comp_leaves_stats[offset].slot_use--;
            comp_leaves_stats[offset].bit_map = bit_map_32bit;
            
            break;

        case 2://64-bit keys
            
            keyInLeaf = (uint64_t) bt_comp_leaves[offset].keys[2 * pos + 1];
            keyInLeaf = (keyInLeaf << 32) | bt_comp_leaves[offset].keys[2 * pos];

            if (keyInLeaf != skey)
                return;

            if (slot_use == 1){
                for (i = 0; i < BLOCKSIZE_32BIT; i++){
                    bt_comp_leaves[offset].keys[i] = UINT_MAX;
                }
                
                comp_leaves_stats[offset].slot_use = 0;
                comp_leaves_stats[offset].bit_map = 0;

                return;
            }

            bit_map_64bit = (uint16_t) comp_leaves_stats[offset].bit_map; 

            uint64_t temp_array_64bit[BLOCKSIZE_64BIT];
            memcpy(&temp_array_64bit, &bt_comp_leaves[offset].keys, BLOCKSIZE_64BIT * sizeof(uint64_t));

            if (pos != BLOCKSIZE_64BIT - 1){
                uint32_t temp_bit_map = bit_map_64bit << (pos + 16);
                valid_pos = _lzcnt_u32(temp_bit_map);   
                
                uint64_t reproducedKey = 0;

                if (valid_pos + pos == BLOCKSIZE_64BIT - 1){
                    reproducedKey = ULONG_MAX;
                }
                else{
                    reproducedKey = temp_array_64bit[pos + 1 + valid_pos];
                }

                for (i = 0; i <= valid_pos; i++){
                    temp_array_64bit[pos + i] = reproducedKey;
                }

                bit_map_64bit = bit_map_64bit ^ (0x8000 >> (pos + valid_pos));
                
            }
            else{
                bit_map_64bit = bit_map_64bit ^ 0x0001;

                valid_pos = _tzcnt_u16(bit_map_64bit);

                for (i = 0; i < valid_pos; i++){
                    temp_array_64bit[pos - i] = ULONG_MAX;
                }

            }
            
            memcpy(&bt_comp_leaves[offset].keys, &temp_array_64bit, BLOCKSIZE_64BIT * sizeof(uint64_t));
            
            comp_leaves_stats[offset].slot_use--;
            comp_leaves_stats[offset].bit_map = bit_map_64bit;
            
            break;
    }
}


//range query for compact bstree
//same code with original bstree, but we have variable length in leaves
uint64_t range_search_compact(uint64_t start_key, uint64_t end_key){
    int64_t count = 0;   
    uint32_t i, pos_start, pos_end, offset = root_offset;
    uint32_t start_range_offset = 0, end_range_offset = 0;
    uint64_t start_skey = start_key;
    uint64_t end_skey = end_key + 1;
    uint64_t keyInLeaf = 0;
    uint8_t leaf_case_start, leaf_case_end;
    
    uint16_t bit_flip_64bit = 0x8000, bit_map_64bit;
    uint32_t bit_flip_32bit = 0x80000000, bit_map_32bit;
    uint64_t bit_flip_16bit = 0x8000000000000000, bit_map_16bit;
    
    //check if the query is wrong
    if (end_key <= start_key)
        return 0;


    // search to find the position of the start_key
    for (i = bt_stats.height; i > 0; i--){
        
        __builtin_prefetch(bt_inners[offset].ptrs);
	    pos_start = successorLinG_compact(bt_inners[offset].keys,start_skey);
        offset = bt_inners[offset].ptrs[pos_start];
    }

    leaf_case_start = comp_leaves_stats[offset].leaf_case;
    pos_start = successorLinGE_compact(bt_comp_leaves[offset].keys, comp_leaves_stats[offset].reference_key, leaf_case_start ,&start_skey);
    start_range_offset = offset;
    
    offset = root_offset;

    //search to find the position of the end_key
    for (i = bt_stats.height; i > 0; i--){
        
        __builtin_prefetch(bt_inners[offset].ptrs);
	    pos_end = successorLinG_compact(bt_inners[offset].keys,end_skey);
        offset = bt_inners[offset].ptrs[pos_end];
    }

    leaf_case_end = comp_leaves_stats[offset].leaf_case;
    pos_end = successorLinGE_compact(bt_comp_leaves[offset].keys, comp_leaves_stats[offset].reference_key, leaf_case_end, &end_skey);
    end_range_offset = offset;

    switch (leaf_case_end){
        case 0: //16-bit differences
            keyInLeaf = (uint64_t) ((bt_comp_leaves[end_range_offset].keys[BLOCKSIZE_32BIT-1] >> 16) & 0x0000ffff);
            break;
        case 1: //32-bit differences
            keyInLeaf = (uint64_t) bt_comp_leaves[end_range_offset].keys[BLOCKSIZE_32BIT - 1];
            break;
        case 2: //64-bit keys
            keyInLeaf = (uint64_t) bt_comp_leaves[end_range_offset].keys[BLOCKSIZE_32BIT - 1];
            keyInLeaf = (keyInLeaf << 32) | bt_comp_leaves[end_range_offset].keys[BLOCKSIZE_32BIT - 2];
            break;
    }

    pos_end = (end_skey > keyInLeaf) ? pos_end + 1 : pos_end;

    // make counting at the first leaf node
    switch (leaf_case_start){
        case 0: //16-bit differences

            keyInLeaf = (uint64_t) ((bt_comp_leaves[start_range_offset].keys[BLOCKSIZE_32BIT-1] >> 16) & 0x0000ffff);

            bit_map_16bit = comp_leaves_stats[start_range_offset].bit_map;
            bit_map_16bit = bit_map_16bit << pos_start;

            if (start_range_offset == end_range_offset){
                if (start_skey > keyInLeaf)
                    return 0;

                if (pos_start == pos_end)
                    return 0;

                for (i = pos_start; i < pos_end; i++){
                    count += ((bit_map_16bit & bit_flip_16bit) >> (BLOCKSIZE_16BIT - 1));
                    bit_map_16bit = bit_map_16bit << 1;
                }   

                return count;
            }
            else{
                if (start_skey > keyInLeaf){
                    pos_start = 0;
                }
                else{
                    for (i = pos_start; i < BLOCKSIZE_16BIT; i++){
                        count += ((bit_map_16bit & bit_flip_16bit) >> (BLOCKSIZE_16BIT - 1));
                        bit_map_16bit = bit_map_16bit << 1;
                    }     
                }
            }
            
            break;

        case 1: //32-bit differences

            keyInLeaf = (uint64_t) bt_comp_leaves[start_range_offset].keys[BLOCKSIZE_32BIT - 1];

            bit_map_32bit = (uint32_t) comp_leaves_stats[start_range_offset].bit_map;
            bit_map_32bit = bit_map_32bit << pos_start;

            if (start_range_offset == end_range_offset){

                if (start_skey > keyInLeaf)
                    return 0;

                if (pos_start == pos_end)
                    return 0;

                for (i = pos_start; i < pos_end; i++){
                    count += ((bit_map_32bit & bit_flip_32bit) >> (BLOCKSIZE_32BIT - 1));
                    bit_map_32bit = bit_map_32bit << 1;
                }   

                return count;
            }
            else{
                if (start_skey > keyInLeaf){
                    pos_start = 0;
                }
                else{
                    for (i = pos_start; i < BLOCKSIZE_32BIT; i++){
                        count += ((bit_map_32bit & bit_flip_32bit) >> (BLOCKSIZE_32BIT - 1));
                        bit_map_32bit = bit_map_32bit << 1;
                    }
                }
            }

            break;
        
        case 2: //64-bit keys

            keyInLeaf = (uint64_t) bt_comp_leaves[start_range_offset].keys[BLOCKSIZE_32BIT - 1];
            keyInLeaf = (keyInLeaf << 32) | bt_comp_leaves[start_range_offset].keys[BLOCKSIZE_32BIT - 2];

            bit_map_64bit = (uint16_t) comp_leaves_stats[start_range_offset].bit_map;
            bit_map_64bit = bit_map_64bit << pos_start;

            if (start_range_offset == end_range_offset){

                if (start_skey > keyInLeaf)
                    return 0;

                if (pos_start == pos_end)
                    return 0;

                for (i = pos_start; i < pos_end; i++){
                    count += ((bit_map_64bit & bit_flip_64bit) >> (BLOCKSIZE_64BIT - 1));
                    bit_map_64bit = bit_map_64bit << 1;
                }   

                return count;
            }
            else{
                if (start_skey > keyInLeaf){
                    pos_start = 0;
                }
                else{
                    for (i = pos_start; i < BLOCKSIZE_64BIT; i++){
                        count += ((bit_map_64bit & bit_flip_64bit) >> (BLOCKSIZE_64BIT - 1));
                        bit_map_64bit = bit_map_64bit << 1;
                    }
                }
            }
            break;
    }

    offset = comp_leaves_stats[start_range_offset].next_leaf;

    //calculate the range query before the last node of the range
    while (offset != end_range_offset){
        uint8_t leaf_case_curr = comp_leaves_stats[offset].leaf_case;

        switch (leaf_case_curr){
            case 0: //16-bit differences

                bit_map_16bit = comp_leaves_stats[offset].bit_map;

                for (i = 0; i < BLOCKSIZE_16BIT; i++){
                    count += (uint16_t) ((bit_map_16bit & bit_flip_16bit) >> (BLOCKSIZE_16BIT - 1));
                    bit_map_16bit = bit_map_16bit << 1;
                }

                break;
            
            case 1: //32-bit differences

                bit_map_32bit = (uint32_t) comp_leaves_stats[offset].bit_map;
                for (i = 0; i < BLOCKSIZE_32BIT; i++){
                    count += ((bit_map_32bit & bit_flip_32bit) >> (BLOCKSIZE_32BIT - 1));
                    bit_map_32bit = bit_map_32bit << 1;
                }

                break;

            case 2: //64-bit keys
                bit_map_64bit = (uint16_t) comp_leaves_stats[offset].bit_map;
                
                for (i = 0; i < BLOCKSIZE_64BIT; i++){
                    count += ((bit_map_64bit & bit_flip_64bit) >> (BLOCKSIZE_64BIT - 1));
                    bit_map_64bit = bit_map_64bit << 1;
                }
                break;
        }

        offset = comp_leaves_stats[offset].next_leaf;
    }

    //last node of the range
    switch (leaf_case_end){
        case 0: //16-bit differences
            bit_map_16bit = comp_leaves_stats[end_range_offset].bit_map;

            for (i = 0; i < pos_end; i++){
                count += (uint16_t) ((bit_map_16bit & bit_flip_16bit) >> (BLOCKSIZE_16BIT - 1));
                bit_map_16bit = bit_map_16bit << 1;
            }

            break;
        
        case 1: //32-bit differences

            bit_map_32bit = (uint32_t) comp_leaves_stats[end_range_offset].bit_map;

            for (i = 0; i < pos_end; i++){
                count += ((bit_map_32bit & bit_flip_32bit) >> (BLOCKSIZE_32BIT - 1));
                bit_map_32bit = bit_map_32bit << 1;
            }

            break;

        case 2: //64-bit keys
            bit_map_64bit = (uint16_t) comp_leaves_stats[end_range_offset].bit_map;
            
            for (i = 0; i < pos_end; i++){
                count += ((bit_map_64bit & bit_flip_64bit) >> (BLOCKSIZE_64BIT - 1));
                bit_map_64bit = bit_map_64bit << 1;
            }
            break;
    }
    
    return count;

}


//delete compact bstree
void delete_tree_compact(){
    free(bt_comp_leaves);
    free(bt_inners);
    cout<<"tree deleted compact"<<endl;
}
#endif