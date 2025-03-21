#ifndef _BSTREEOLC_H_
#define _BSTREEOLC_H_


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
#include "defOLC.h"
#include <atomic>
#include <sched.h>


void yield(int count) {
    if (count>3)
      sched_yield();
    else
      _mm_pause();
  }

void print_leaves()
{
    for (uint32_t i = 0; i < bt_stats.num_leaves; i++)
    {   
        printf("For Leaf Node %u: \n", i);
        printf("- Keys:\n");
        for (uint32_t j = 0; j < BLOCKSIZE; j++)
        {
            printf("%lu ", bt_leaves[i].keys[j]);
        }
        printf("\n- Statistics:\n");
        printf("Slotuse = %d -- Bitmap = %x -- nextLeaf = %u\n\n", leaves_stats[i].slot_use, leaves_stats[i].bit_map, leaves_stats[i].next_leaf);
    }
}

void print_inners()
{
    for (uint32_t i = 0; i < bt_stats.num_inners; i++)
    {   
        printf("For Inner Node %u: \n", i);
        printf("- Keys:\n");
        for (uint32_t j = 0; j < BLOCKSIZE; j++)
        {
            printf("%lu ", bt_inners[i].keys[j]);
        }
        
        printf("\n- Pointers:\n");
        for (uint32_t j = 0; j < BLOCKSIZE; j++)
        {
            printf("%u ", bt_inners[i].ptrs[j]);
        }

        printf("\n- Statistics:\n");
        printf("Slotuse = %d -- Bitmap = %x\n\n", inners_stats[i].slot_use, inners_stats[i].bit_map);
    }
}


void memory_footprint()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        std::cout << "Maximum Resident Set Size: " << usage.ru_maxrss << " KB" << std::endl;
        // Other fields can be printed similarly if needed
    } else {
        std::cerr << "Error in getrusage" << std::endl;
    }
}

void reallocate_leaves(){
    uint32_t oldSize = bt_stats.max_num_leaves.load();
    uint32_t newSize = oldSize * 2;

    btree_leaf *old_ptr = bt_leaves.load();
    btree_leaf *new_ptr;
    stats_leaf *old_stats = leaves_stats.load(), *new_stats;


    new_ptr = (btree_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(btree_leaf));
    
    if (new_ptr == NULL)
    {
        printf("BAD ALLOCATION AT LEAF NODES\n");
        exit(-1);
    }

    new_stats = (stats_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(stats_leaf));

    if (new_stats == NULL)
    {
        printf("BAD ALLOCATION AT LEAF STATS\n");
        exit(-1);
    }

    if (old_ptr)
    {
        memcpy(new_ptr, old_ptr, oldSize * sizeof(btree_leaf));
    }

    if (old_stats)
    {
        memcpy(new_stats, old_stats, oldSize * sizeof(stats_leaf));
    }


    if (bt_leaves.compare_exchange_strong(old_ptr, new_ptr))
    {
        leaves_stats.store(new_stats);
        bt_stats.max_num_leaves.store(newSize);

        free(old_ptr);
        free(old_stats);
    }
    else
    {
        free(new_ptr);
        free(new_stats);
    }

}

void reallocate_inners(){
    uint32_t oldSize = bt_stats.max_num_inners.load();
    uint32_t newSize = oldSize * 2;    

    uint32_t expectedSize = oldSize;

    if (!bt_stats.max_num_inners.compare_exchange_strong(expectedSize, newSize))
        return;

    btree_inner *old_ptr = bt_inners.load();
    btree_inner *new_ptr;
    stats_inner *old_stats = inners_stats.load(), *new_stats;

    uint32_t num_pages_inner = ((newSize * sizeof(btree_inner) -1) / page_size) + 1;//initialize num pages for inner nodes
    new_ptr = (btree_inner *) aligned_alloc(page_size, num_pages_inner * page_size);
    madvise(new_ptr, num_pages_inner*page_size, MADV_HUGEPAGE);

    if (new_ptr == NULL)
    {
        printf("BAD ALLOCATION AT INNER NODES\n");
        exit(-1);
    }

    new_stats = (stats_inner *) aligned_alloc(64, newSize * sizeof(stats_inner));

    if (new_stats == NULL)
    {
        printf("BAD ALLOCATION AT INNER STATS\n");
        exit(-1);
    }

    // if (old_ptr)
    // {
    memcpy(new_ptr, old_ptr, oldSize * sizeof(btree_inner));
    // }
        

    // if (old_stats)
    // {
    memcpy(new_stats, old_stats, oldSize * sizeof(stats_inner));
    // }

    bt_inners.store(new_ptr);
    inners_stats.store(new_stats);
    bt_stats.max_num_inners.store(newSize);

    free(old_ptr);
    free(old_stats);

    // if (bt_inners.compare_exchange_strong(old_ptr, new_ptr))
    // {
    //     inners_stats.store(new_stats);
    //     bt_stats.max_num_inners.store(newSize);

    //     free(old_ptr);
    //     free(old_stats);
    // }
    // else
    // {
    //     free(new_ptr);
    //     free(new_stats);
    // }

}
inline uint32_t successorLinG_cnt(uint64_t *ar, uint64_t key)
{
	uint32_t count = 0;
	
	for(int i=0;i<BLOCKSIZE;i++)
		count += key>=ar[i]; // evaluates to 0 if false, evaluates to 1 if true

	return count;
}

// finds lowest position greater than or equal to key with count
inline uint32_t successorLinGE_cnt(uint64_t *ar, uint64_t key)
{
	int count = 0;
	
	for(int i=0;i<BLOCKSIZE;i++)
		count += key>ar[i]; // evaluates to 0 if false, evaluates to 1 if true

	return count;
}


// finds lowest position greater than key with count
inline uint32_t successorLinG(uint64_t *ar, uint64_t key)
{
	uint32_t s1 = 0;

	__m512i threshold_vector = _mm512_set1_epi64(key);

    __m512i y1 = _mm512_loadu_epi64((__m512i*) ar);
    // __m512i y1 = _mm512_load_epi64((__m512i*) ar);
    uint32_t m1 = (uint32_t) _mm512_cmpge_epu64_mask(threshold_vector,y1);
    s1 = _mm_popcnt_u32(m1);

    __m512i y2 = _mm512_loadu_epi64((__m512i*)(ar + 8)); 
    // __m512i y2 = _mm512_load_epi64((__m512i*)(ar + 8)); 
    uint32_t m2 = (uint32_t) _mm512_cmpge_epu64_mask(threshold_vector,y2);
    s1 += _mm_popcnt_u32(m2);

	return s1;
}

// finds lowest position greater than or equal to key with count
inline uint32_t successorLinGE(uint64_t *ar, uint64_t key)
{
	uint32_t s1 = 0;

    __m512i threshold_vector = _mm512_set1_epi64(key);
	
    __m512i y1 = _mm512_loadu_epi64((__m512i*) ar);
    // __m512i y1 = _mm512_load_epi64((__m512i*) ar);
    uint32_t m1 = (uint32_t) _mm512_cmpgt_epu64_mask(threshold_vector, y1);
    s1 = _mm_popcnt_u32(m1);

    __m512i y2 = _mm512_loadu_epi64((__m512i*) (ar + 8)); 
    // __m512i y2 = _mm512_load_epi64((__m512i*) (ar + 8)); 
    uint32_t m2 = (uint32_t) _mm512_cmpgt_epu64_mask(threshold_vector, y2);
    s1 += _mm_popcnt_u32(m2);
    
    s1 = (s1 == BLOCKSIZE) ? --s1 : s1;

    return s1;
}

uint64_t *construct_leaves(uint64_t *ar, uint64_t size_data)
{   
    uint32_t i = 0, j = 0;
    uint64_t limited_size = 3 << 30;
    uint64_t NUMnextlevelnonleafkeys = 0, prev_inserted_leaf = 0;    
    uint64_t * nextlevelnonleafkeys;


    bt_stats.num_leaves = ceil((double)size_data / KEYS_IN_LEAF);   //initialize num leaves based on the num of gaps   
    bt_stats.max_num_leaves = 2 * bt_stats.num_leaves;

    if (bt_stats.max_num_leaves * sizeof(btree_leaf) <= limited_size)
    {
        uint32_t num_pages_leaves = ((bt_stats.max_num_leaves * sizeof(btree_leaf) -1) / page_size) + 1;

        bt_leaves = (btree_leaf *) aligned_alloc(page_size,  num_pages_leaves*page_size);//allocation memory for leaves
        madvise(bt_leaves, num_pages_leaves * page_size, MADV_HUGEPAGE);
    }
    else
    {
        bt_leaves = (btree_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(btree_leaf));
    }

    leaves_stats = (stats_leaf *) aligned_alloc(64,bt_stats.max_num_leaves*sizeof(stats_leaf));   // allocate the additional space for the statistics of the leaves
    
    
    // uint32_t num_pages_for_stats = ((bt_stats.max_num_leaves * sizeof(stats_leaf) -1) / page_size);
    // leaves_stats = (stats_leaf *) aligned_alloc(page_size, num_pages_for_stats * page_size);   // allocate the additional space for the statistics of the leaves
    // madvise(leaves_stats, num_pages_for_stats * page_size, MADV_HUGEPAGE);
    
    nextlevelnonleafkeys = (uint64_t *)aligned_alloc(64, sizeof(uint64_t) * bt_stats.num_leaves);   // keys (separators) stored in non-leaf nodes (next level)

    //create all node leave, except the last one
    for (i = 0; prev_inserted_leaf + KEYS_IN_LEAF < size_data; i++)
    {
        uint64_t diff = ar[prev_inserted_leaf + BLOCKSIZE -1] - ar[prev_inserted_leaf];

        //sequential keys
        if (diff == BLOCKSIZE - 1){
            leaves_stats[i].slot_use = BLOCKSIZE;
            leaves_stats[i].bit_map = 0xffff;

            memcpy(bt_leaves[i].keys, &ar[prev_inserted_leaf], BLOCKSIZE * sizeof(uint64_t));
            prev_inserted_leaf += BLOCKSIZE;
         
        }
        else{//non-sequential keys
            leaves_stats[i].slot_use = KEYS_IN_LEAF;
            leaves_stats[i].bit_map = 0xeeee;
            
            memcpy(&bt_leaves[i].keys, &ar[prev_inserted_leaf], STEP_LEAF * sizeof(uint64_t));
            prev_inserted_leaf += STEP_LEAF;

            //first gap
            bt_leaves[i].keys[3] = ar[prev_inserted_leaf];
            
            memcpy(&bt_leaves[i].keys[4], &ar[prev_inserted_leaf], STEP_LEAF * sizeof(uint64_t));
            prev_inserted_leaf += STEP_LEAF;

            //second gap
            bt_leaves[i].keys[7] = ar[prev_inserted_leaf];
            
            memcpy(&bt_leaves[i].keys[8], &ar[prev_inserted_leaf], STEP_LEAF * sizeof(uint64_t));
            prev_inserted_leaf += STEP_LEAF;
            
            //third gap
            bt_leaves[i].keys[11] = ar[prev_inserted_leaf];
            
            memcpy(&bt_leaves[i].keys[12], &ar[prev_inserted_leaf], STEP_LEAF * sizeof(uint64_t));
            prev_inserted_leaf += STEP_LEAF;

            bt_leaves[i].keys[BLOCKSIZE - 1] = ULONG_MAX;
        }

       leaves_stats[i].next_leaf = i+1;
        if (i>0) nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = bt_leaves[i].keys[0];
    }

    // last leaf
    int residue = (size_data - prev_inserted_leaf);
   
    uint16_t temp_bit_map = 0xffff;
    temp_bit_map = temp_bit_map << (BLOCKSIZE - residue);
    
    leaves_stats[i].slot_use = residue;
    leaves_stats[i].bit_map = temp_bit_map;
                
    
    memcpy(&bt_leaves[i].keys,&ar[prev_inserted_leaf],residue*sizeof(uint64_t));
    for (j = residue;j<BLOCKSIZE;j++){
        bt_leaves[i].keys[j] = ULONG_MAX;
        
    }
    nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = bt_leaves[i].keys[0];
    leaves_stats[i].next_leaf = 0;
    
    bt_stats.num_leaves = i + 1;


    return nextlevelnonleafkeys;
}

void construct_inners(uint64_t *nextlevelnonleafkeys)
{
    uint64_t i = 0, j = 0;
    uint64_t NUMnextlevelnonleafkeys = bt_stats.num_leaves;
    uint64_t prev_inserted_inner = 0;
    uint32_t nextpointedblockid = 0; 
    uint32_t previous_level = 0;
    uint32_t numcblocks =  0; 
    uint32_t idx = 0;

    uint32_t levels_without_gaps = 1;
    uint32_t height_16 = ceil(log2(bt_stats.num_leaves.load())/log2(BLOCKSIZE));
    uint32_t height_15 = ceil(log2(bt_stats.num_leaves.load())/log2(KEYS_IN_INNER + 1));
    

    if (height_16 == height_15){
        bt_stats.height = height_15;
    }
    else{
        bt_stats.height = height_16;
        levels_without_gaps = floor(height_15/2);
    }

    uint32_t lut[bt_stats.height.load()]; // lookup table initialization 

    for (i = 0; i < bt_stats.height-levels_without_gaps; i++){
        lut[i] = ceil((double)NUMnextlevelnonleafkeys/(KEYS_IN_INNER + 1));

        bt_stats.num_inners += lut[i];

        NUMnextlevelnonleafkeys = lut[i];
    }

    for (; i < bt_stats.height-1; i++){// find num of inner nodes and keep the size of each table in the lookup table 

        lut[i] = ceil((double)NUMnextlevelnonleafkeys/(BLOCKSIZE));

        bt_stats.num_inners += lut[i];

        NUMnextlevelnonleafkeys = lut[i];
    }

    lut[bt_stats.height - 1] = 1; // root always has one node

    bt_stats.num_inners += 1;
    bt_stats.max_num_inners =  bt_stats.max_num_leaves / 10; //initialize max num inner nodes
    uint32_t num_pages_inner = ((bt_stats.max_num_inners * sizeof(btree_inner) -1) / page_size) + 1;//initialize num pages for inner nodes

    bt_inners = (btree_inner *)aligned_alloc(page_size,  num_pages_inner*page_size); //allocation memory for inner
    madvise(bt_inners, num_pages_inner*page_size, MADV_HUGEPAGE);

    //bt_inners = (btree_inner *)aligned_alloc(64,  bt_stats.max_num_inners * sizeof(btree_inner)); //allocation memory for inner

    inners_stats  = (stats_inner *) aligned_alloc(64, bt_stats.max_num_inners * sizeof(stats_inner)); //allocate additional space for the statistics of the inners

    NUMnextlevelnonleafkeys = bt_stats.num_leaves-1;

    if (NUMnextlevelnonleafkeys)
	{	
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
                for (ushort t= STEP_INNER + 1; t<=BLOCKSIZE -1;t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += STEP_INNER;
                
                bt_inners[idx + i].keys[BLOCKSIZE-1] = ULONG_MAX;

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
                for (ushort t= STEP_INNER + 1; t<=BLOCKSIZE -1;t++){
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid;
                    nextpointedblockid++;
                }
                prev_inserted_inner += STEP_INNER;
                
                bt_inners[idx + i].keys[BLOCKSIZE-1] = ULONG_MAX;

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
                for (ushort t = residue; t < BLOCKSIZE; t++){
                    bt_inners[idx + i].keys[t] = ULONG_MAX;
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid ;                    

                    if ( t < BLOCKSIZE -1 ){
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
	  		    for (;j<BLOCKSIZE;j++){
		  		    bt_inners[idx + i].keys[j] =  ULONG_MAX;
                    bt_inners[idx + i].ptrs[j] = nextpointedblockid ;
                }
            }
    
            nextpointedblockid = idx;
        }

        for (; level < bt_stats.height - 1 ; level ++){
            prev_inserted_inner = 0;
			
			uint32_t numptrs = NUMnextlevelnonleafkeys + 1; // include leftmost pointer which has no key
			NUMnextlevelnonleafkeys = 0;
					
            numcblocks = lut[level];
		    idx = 0;
            for (j = level + 1; j < bt_stats.height ; j ++){// calculate the position that each level will be saved in the array of inner nodes. bottom-up implementation
                idx += lut[j];// idx is used to write inner nodes to the appropriate position to the inner array from right to left(first left position is root)
            }

			
			for (i=0; i<numcblocks-1; i++) {

                memcpy(&bt_inners[idx + i].keys, &nextlevelnonleafkeys[prev_inserted_inner], (BLOCKSIZE - 1) * sizeof(uint64_t));
                bt_inners[idx + i].keys[BLOCKSIZE - 1] = ULONG_MAX;
                
                inners_stats[idx+i].slot_use = BLOCKSIZE-1;
                inners_stats[idx+i].bit_map = 0xffff;

                for (uint32_t t = 0; t < BLOCKSIZE; t++)
                {
                    bt_inners[idx + i].ptrs[t] = nextpointedblockid++;
                }
                prev_inserted_inner += BLOCKSIZE -1;
                //update parent array for the next level
                nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = nextlevelnonleafkeys[prev_inserted_inner];
                prev_inserted_inner++;          
			}
            

			uint32_t residue = numptrs%BLOCKSIZE;
            uint16_t temp_bit_map = 0xffff;
            if(residue == 0) residue = BLOCKSIZE;

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
            for (;j<BLOCKSIZE;j++){
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
        temp_bit_map = temp_bit_map << (BLOCKSIZE - NUMnextlevelnonleafkeys);
        temp_bit_map = temp_bit_map | 0x1;

        inners_stats[0].slot_use = NUMnextlevelnonleafkeys;
        inners_stats[0].bit_map = temp_bit_map;

        memcpy(&bt_inners[0].keys,nextlevelnonleafkeys,NUMnextlevelnonleafkeys*sizeof(uint64_t));
        
        for (j=0; j<=NUMnextlevelnonleafkeys;j++){
            bt_inners[0].ptrs[j] = nextpointedblockid;
            nextpointedblockid ++ ;
        }

        nextpointedblockid--;
        for (j = NUMnextlevelnonleafkeys;j<BLOCKSIZE;j++){
            bt_inners[0].keys[j] = ULONG_MAX;
            bt_inners[0].ptrs[j] = nextpointedblockid;
        }        
	}
}


bool search(uint64_t skey){

    int restartCount = 0;
    
    restart:
            if (restartCount++)
                yield(restartCount);
        
            bool needRestart = false;

            uint32_t i, pos, offset = root_offset;

            uint64_t versionNode = bt_inners[offset].readLockOrRestart(needRestart);
        
            if(needRestart || (offset != root_offset))
            {    
                // printf("offset(1) = %lu -- skey = %lu\n", offset, skey);
                goto restart;
            }    

            int64_t parent_offset = -1;
            uint64_t versionParent; 

            for (i = bt_stats.height; i > 1; i--)
            {

                if (parent_offset >= 0)
                {
                    bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);
                    if (needRestart)
                    {
                        // printf("parentOffset(1) = %lu -- skey = %lu\n", parent_offset, skey);
                        goto restart;
                    }
                }

                parent_offset = offset;
                versionParent = versionNode;

                __builtin_prefetch(bt_inners[offset].ptrs);
                pos = successorLinG(bt_inners[offset].keys,skey); 
                //pos = successorLinG_cnt(bt_inners[offset].keys,skey);

                bt_inners[parent_offset].checkOrRestart(versionNode, needRestart);

                if (needRestart)
                {
                    // printf("parentOffset(2) = %lu -- skey = %lu\n", parent_offset, skey);
                    goto restart;
                }

                offset = bt_inners[offset].ptrs[pos];
                versionNode = bt_inners[offset].readLockOrRestart(needRestart);

                if (needRestart)
                {
                    // printf("offset(2) = %lu -- skey = %lu\n", offset, skey);
                    goto restart;
                }

            }


            if (parent_offset >= 0)
            {
                bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);
                if (needRestart)
                {
                    // printf("parentOffset(3) = %lu -- skey = %lu\n", parent_offset, skey);
                    goto restart;
                }
            }

            parent_offset = offset;
            versionParent = versionNode;

            __builtin_prefetch(bt_inners[offset].ptrs);
            pos = successorLinG(bt_inners[offset].keys,skey); 
            //pos = successorLinG_cnt(bt_inners[offset].keys,skey);

            bt_inners[parent_offset].checkOrRestart(versionNode, needRestart);

            if (needRestart)
            {
                // printf("parentOffset(4) = %lu -- skey = %lu\n", parent_offset, skey);
                goto restart;
            }

            offset = bt_inners[offset].ptrs[pos];
            versionNode = bt_leaves[offset].readLockOrRestart(needRestart);

            if (needRestart)
            {
                // printf("offset(3) = %lu -- skey = %lu\n", offset, skey);
                goto restart;
            }

            bool success;
            pos = successorLinGE(bt_leaves[offset].keys,skey);
            //pos = successorLinGE_cnt(bt_leaves[offset].keys,skey);

            if (bt_leaves[offset].keys[pos]==skey)
                success = true;

            if (parent_offset >= 0)
            {
                bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);

                if (needRestart)
                {
                    // printf("parentOffset(5) = %lu -- skey = %lu\n", parent_offset, skey);
                    goto restart;
                }
            }

            bt_leaves[offset].readUnlockOrRestart(versionNode, needRestart);

            if(needRestart)
            {
                // printf("offset(4) = %lu -- skey = %lu\n", offset, skey);
                goto restart;
            }

        return success;
}

void insert(uint64_t ikey)
{
    int restartCountSearch;

    restartSearch:
        if (restartCountSearch++)
        {
            yield(restartCountSearch);
        }
        bool needRestart = false;

        uint32_t i, pos, offset = root_offset;

        uint64_t versionNode = bt_inners[offset].readLockOrRestart(needRestart);
    
        if(needRestart || (offset != root_offset))
        {    
            goto restartSearch;
        }

        int64_t parent_offset = -1;
        uint64_t versionParent;

        for (i = bt_stats.height; i > 1; i--)
        {
            if (parent_offset >= 0)
            {
                bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);
                if (needRestart)
                {
                    goto restartSearch;
                }
            }

            parent_offset = offset;
            versionParent = versionNode;

            __builtin_prefetch(bt_inners[offset].ptrs);
            pos = successorLinG(bt_inners[offset].keys,ikey); 

            bt_inners[parent_offset].checkOrRestart(versionNode, needRestart);

            if (needRestart)
            {
                goto restartSearch;
            }

            offset = bt_inners[offset].ptrs[pos];
            versionNode = bt_inners[offset].readLockOrRestart(needRestart);

            if (needRestart)
            {
                goto restartSearch;
            }
        }

        if (parent_offset >= 0)
        {
            bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);
            if (needRestart)
            {
                goto restartSearch;
            }
        }

        parent_offset = offset;
        versionParent = versionNode;

        __builtin_prefetch(bt_inners[offset].ptrs);
        pos = successorLinG(bt_inners[offset].keys,ikey); 

        bt_inners[parent_offset].checkOrRestart(versionNode, needRestart);

        if (needRestart)
        {
            goto restartSearch;
        }

        offset = bt_inners[offset].ptrs[pos];
        versionNode = bt_leaves[offset].readLockOrRestart(needRestart);

        if (needRestart)
        {
            goto restartSearch;
        }

        uint8_t slotuse = leaves_stats[offset].slot_use;
        if (slotuse < BLOCKSIZE)    // Leaf has empty slot for insertion
        {

            bt_leaves[offset].upgradeToWriteLockOrRestart(versionNode, needRestart);

            if (needRestart)
            {
                goto restartSearch;
            }

            bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);

            if (needRestart)
            {
                bt_leaves[offset].writeUnlock();
                goto restartSearch;
            }
        
            pos = successorLinGE(bt_leaves[offset].keys, ikey);
        

            if (bt_leaves[offset].keys[pos] == ikey)
            {
                bt_leaves[offset].writeUnlock();
                return;
            }

            uint16_t bit_map = ~leaves_stats[offset].bit_map;

            if (slotuse != 0)
            {
                uint32_t shift_pos = 0;
    
                if (pos == 0)
                {
                    shift_pos = _lzcnt_u32((uint32_t)bit_map);
                    
                    if (shift_pos > 16){
                        uint16_t bit_flip = 0x8000;
                        uint32_t diff = shift_pos - 16;
    
                        memcpy(&bt_leaves[offset].keys[pos + 1], &bt_leaves[offset].keys[pos] , diff * sizeof(uint64_t));
                        bit_map = bit_map ^ (bit_flip >> diff);
                    }
                    else{
                        bit_map = bit_map ^ 0x8000;
                    }
                    
                    bt_leaves[offset].keys[pos] = ikey;
                
                }
                else if (pos == BLOCKSIZE - 1)
                {
                    shift_pos = _tzcnt_u16(bit_map);
                
                    if (shift_pos > 0){
                        uint16_t bit_flip = 0x0001;
                        uint32_t start_copy = pos - shift_pos;
                        uint32_t keysForCopy = shift_pos;
    
                        if (ikey < bt_leaves[offset].keys[BLOCKSIZE - 1])
                        {
                            keysForCopy--;
                            pos--;
                        }
    
                        memcpy(&bt_leaves[offset].keys[start_copy], &bt_leaves[offset].keys[start_copy + 1] , keysForCopy * sizeof(uint64_t) );
                        bit_map = bit_map ^ (bit_flip << shift_pos);
                        bt_leaves[offset].keys[pos] = ikey;
                    }
                    else{
                        bit_map = bit_map ^ 0x0001;
                        bt_leaves[offset].keys[pos] = ikey;
                    }
                }
                else
                {
    
                    if (bt_leaves[offset].keys[pos] == bt_leaves[offset].keys[pos + 1])
                    {
                        bt_leaves[offset].keys[pos] = ikey;
                        bit_map = bit_map ^ (0x8000 >> pos);
    
                    }
                    else
                    {
                        uint32_t temp_bit_map = bit_map << (pos + 16);
                        shift_pos = _lzcnt_u32(temp_bit_map);
    
                        if (shift_pos == 32){
                            temp_bit_map = bit_map >> (16 - pos - 1); 
                            shift_pos =_tzcnt_u32(temp_bit_map);
    
                            uint16_t bit_flip = 0x8000;
                            uint32_t start_copy = pos - shift_pos;
                            
                            memcpy(&bt_leaves[offset].keys[start_copy], &bt_leaves[offset].keys[start_copy + 1] , (shift_pos-1) * sizeof(uint64_t) );
                            bit_map = bit_map ^ (bit_flip >> start_copy);
                            bt_leaves[offset].keys[pos-1] = ikey;
                        }
                        else{    
                            uint16_t bit_flip = 0x8000;
                            uint32_t diff = shift_pos;// - 16;
                            
                            memcpy(&bt_leaves[offset].keys[pos + 1], &bt_leaves[offset].keys[pos] , diff * sizeof(uint64_t) );
                            bit_map = bit_map ^ (bit_flip >> (pos + diff));
                            bt_leaves[offset].keys[pos] = ikey;
                        }
                    }
                }
            }
            else
            {
                bt_leaves[offset].keys[0] = ikey;
                bit_map = bit_map ^ 0x8000;
            }

            leaves_stats[offset].slot_use++;
            leaves_stats[offset].bit_map = ~bit_map;

            bt_leaves[offset].writeUnlock();
            
            return;
        }

        /* Leaf Node is full, splitting is needed */

        bt_inners[parent_offset].upgradeToWriteLockOrRestart(versionParent, needRestart);

        if (needRestart)
        {
            // bt_leaves[offset].writeUnlock();
            goto restartSearch;
        }

        bt_leaves[offset].upgradeToWriteLockOrRestart(versionNode, needRestart);
        
        if (needRestart)
        {        
            bt_inners[parent_offset].writeUnlock();
            goto restartSearch;
        }

        // if (parent_offset < 0 && (offset != root_offset))
        // {
        //     bt_leaves[offset].writeUnlock();
        //     goto restartSearch;
        // }
        

        pos = successorLinGE(bt_leaves[offset].keys, ikey);

        if (bt_leaves[offset].keys[pos] == ikey)
        {
            bt_leaves[offset].writeUnlock();
            bt_inners[parent_offset].writeUnlock();

            return;
        }

        uint32_t new_offset = bt_stats.num_leaves.fetch_add(1);

        if (new_offset >= bt_stats.max_num_leaves.load())
        {
            reallocate_leaves();
        }

        uint64_t temp_keys[BLOCKSIZE + 1];
        uint32_t mid_in_node = BLOCKSIZE >> 1;
        uint64_t keyNextLevel = 0;
        uint32_t ptrNextLevel = 0;

        memcpy(&temp_keys, &bt_leaves[offset].keys, pos * sizeof(uint64_t));

        if (ikey > bt_leaves[offset].keys[BLOCKSIZE - 1])
        {
            temp_keys[pos] = bt_leaves[offset].keys[BLOCKSIZE - 1];
            temp_keys[pos + 1] = ikey;
        }
        else
        {
            temp_keys[pos] = ikey;
            memcpy(&temp_keys[pos + 1], &bt_leaves[offset].keys[pos], (BLOCKSIZE - pos) * sizeof(uint64_t));
        }

        keyNextLevel = temp_keys[mid_in_node + 1];
        ptrNextLevel = new_offset;

        bt_leaves[offset].keys[0] = temp_keys[0];
        bt_leaves[new_offset].keys[0] = keyNextLevel;


        uint32_t j = 1;
        for (uint32_t i = 1; i < BLOCKSIZE - 1; i += 2, j++)
        {
            bt_leaves[offset].keys[i] = temp_keys[j];
            bt_leaves[offset].keys[i + 1] = temp_keys[j];
    
            bt_leaves[new_offset].keys[i] = temp_keys[mid_in_node + j + 1];
            bt_leaves[new_offset].keys[i + 1] = temp_keys[mid_in_node + j + 1];
        }

        bt_leaves[offset].keys[BLOCKSIZE - 1] = temp_keys[mid_in_node];
        bt_leaves[new_offset].keys[BLOCKSIZE - 1] = ULONG_MAX;

        leaves_stats[new_offset].slot_use = mid_in_node;
        leaves_stats[new_offset].bit_map = 0xaaaa;
        leaves_stats[new_offset].next_leaf = leaves_stats[offset].next_leaf;

        leaves_stats[offset].slot_use = mid_in_node + 1;
        leaves_stats[offset].bit_map = 0xaaab;
        leaves_stats[offset].next_leaf = new_offset;

        bt_leaves[offset].writeUnlock();

        uint32_t currentDepth = 1;
        int restartCountSplit = 0;


        __builtin_prefetch(bt_inners[parent_offset].ptrs);
        pos = successorLinG(bt_inners[parent_offset].keys,keyNextLevel);

        if (inners_stats[parent_offset].slot_use < BLOCKSIZE - 1)
        {
            uint32_t shift_pos = 0;
            uint16_t bit_map = ~inners_stats[parent_offset].bit_map;

            if (bt_inners[parent_offset].keys[pos] == bt_inners[parent_offset].keys[pos + 1]){
                //update key
                bt_inners[parent_offset].keys[pos] = keyNextLevel;

                bit_map = bit_map ^ (0x8000 >> pos);

                if (bt_inners[parent_offset].keys[pos + 1] != ULONG_MAX)
                {
                    bt_inners[parent_offset].ptrs[pos + 1] = ptrNextLevel;
                }
                else
                {
                    for (uint32_t p = pos + 1; p < BLOCKSIZE; p++)
                    {
                        bt_inners[parent_offset].ptrs[p] = ptrNextLevel;
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
                    memcpy(&bt_inners[parent_offset].keys[start_copy], &bt_inners[parent_offset].keys[start_copy + 1] , (shift_pos-1) * sizeof(uint64_t) );
                    bit_map = bit_map ^ (bit_flip >> start_copy);
                    bt_inners[parent_offset].keys[pos-1] = keyNextLevel;

                    //update pointers
                    memcpy(&bt_inners[parent_offset].ptrs[start_copy], &bt_inners[parent_offset].ptrs[start_copy + 1] , shift_pos* sizeof(uint32_t) );
                    bt_inners[parent_offset].ptrs[pos] = ptrNextLevel;
                }
                else{

                    //update keys
                    uint16_t bit_flip = 0x8000;
                    uint32_t diff = shift_pos;// - 16;
                    memcpy(&bt_inners[parent_offset].keys[pos + 1], &bt_inners[parent_offset].keys[pos] , diff * sizeof(uint64_t) );
                    bit_map = bit_map ^ (bit_flip >> (pos + diff));
                    bt_inners[parent_offset].keys[pos] = keyNextLevel;
                    
                    memcpy(&bt_inners[parent_offset].ptrs[pos + 1], &bt_inners[parent_offset].ptrs[pos] , diff  * sizeof(uint32_t) );
                    bt_inners[parent_offset].ptrs[pos+1] = ptrNextLevel;
                }
            }

            inners_stats[parent_offset].slot_use++;
            inners_stats[parent_offset].bit_map = ~bit_map;

            bt_inners[parent_offset].writeUnlock();

            return;
        }
        else    // split at inner node is needed
        {
            uint32_t new_offset = bt_stats.num_inners.fetch_add(1);

            if (new_offset >= bt_stats.max_num_inners.load())
            {
                reallocate_inners();
            }

            uint64_t temp_keys[BLOCKSIZE];
            uint32_t temp_ptrs[BLOCKSIZE + 1];
            mid_in_node = BLOCKSIZE/2;


            memcpy(&temp_keys, &bt_inners[parent_offset].keys, pos * sizeof(uint64_t));
            temp_keys[pos] = keyNextLevel;

            memcpy(&temp_ptrs, &bt_inners[parent_offset].ptrs, (pos + 1) * sizeof(uint32_t));
            temp_ptrs[pos + 1] = ptrNextLevel;

            if (keyNextLevel < bt_inners[parent_offset].keys[BLOCKSIZE - 2])
            {
                memcpy(&temp_keys[pos + 1], &bt_inners[parent_offset].keys[pos], (BLOCKSIZE - pos - 1) * sizeof(uint64_t));
                memcpy(&temp_ptrs[pos + 2], &bt_inners[parent_offset].ptrs[pos + 1], (BLOCKSIZE - pos - 1) * sizeof(uint32_t));
            }

            bt_inners[parent_offset].keys[0] = temp_keys[0];
            bt_inners[parent_offset].ptrs[0] = temp_ptrs[0];

            uint32_t k = 1;
            for (i = 1; i < BLOCKSIZE - 1; i += 2, k++)
            {
                bt_inners[parent_offset].keys[i] = temp_keys[k];
                bt_inners[parent_offset].keys[i + 1] = temp_keys[k];

                bt_inners[parent_offset].ptrs[i] = temp_ptrs[k];
                bt_inners[parent_offset].ptrs[i + 1] = temp_ptrs[k];
            }

            bt_inners[parent_offset].keys[BLOCKSIZE - 1] = ULONG_MAX;
            bt_inners[parent_offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];

            keyNextLevel = temp_keys[k++];
            ptrNextLevel = new_offset;



            for (i = 0; i < BLOCKSIZE - 2; i += 2, k++)
            {
                bt_inners[new_offset].keys[i] = temp_keys[k];
                bt_inners[new_offset].keys[i + 1] = temp_keys[k];

                bt_inners[new_offset].ptrs[i] = temp_ptrs[k];
                bt_inners[new_offset].ptrs[i + 1] = temp_ptrs[k];
            }

            bt_inners[new_offset].keys[BLOCKSIZE - 2] = ULONG_MAX;
            bt_inners[new_offset].keys[BLOCKSIZE - 1] = ULONG_MAX;

            bt_inners[new_offset].ptrs[BLOCKSIZE - 2] = temp_ptrs[k];
            bt_inners[new_offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];

            //slotuse and bitmaps for previous node
            inners_stats[parent_offset].slot_use = mid_in_node;
            inners_stats[parent_offset].bit_map = 0xaaab;

            //slotuse and bitmaps for new node
            inners_stats[new_offset].slot_use = mid_in_node-1;
            inners_stats[new_offset].bit_map = 0x5555;

            bt_inners[parent_offset].writeUnlock();

            currentDepth = 2;

            goto restartSplit;
        }

        
    restartSplit:

        if (restartCountSplit++)
            yield(restartCountSplit);

        needRestart = false;

        offset = root_offset;

        versionNode = bt_inners[offset].readLockOrRestart(needRestart);

        if(needRestart || (offset != root_offset))
        {    
            goto restartSplit;
        }    

        parent_offset = -1;
        
        for (i = bt_stats.height; i > currentDepth; i--)
        {
            if (parent_offset >= 0)
            {
                bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);
                if (needRestart)
                {
                    goto restartSplit;
                }
            }

            parent_offset = offset;
            versionParent = versionNode;

            __builtin_prefetch(bt_inners[offset].ptrs);
            pos = successorLinG(bt_inners[offset].keys,keyNextLevel); 

            bt_inners[parent_offset].checkOrRestart(versionNode, needRestart);

            if (needRestart)
            {
                goto restartSplit;
            }

            offset = bt_inners[offset].ptrs[pos];
            versionNode = bt_inners[offset].readLockOrRestart(needRestart);

            if (needRestart)
            {
                goto restartSplit;
            }
        }


        if (offset != root_offset)
        {
            if (inners_stats[offset].slot_use < BLOCKSIZE - 1)  // lock only this node for insertion
            {

                bt_inners[offset].upgradeToWriteLockOrRestart(versionNode, needRestart);

                if (needRestart)
                {
                    goto restartSplit;
                }
                
                bt_inners[parent_offset].readUnlockOrRestart(versionParent, needRestart);

                if (needRestart)
                {
                    bt_inners[offset].writeUnlock();
                    goto restartSplit;
                }


                __builtin_prefetch(bt_inners[offset].ptrs);
                pos = successorLinG(bt_inners[offset].keys,keyNextLevel);


                uint32_t shift_pos = 0;
                uint16_t bit_map = ~inners_stats[offset].bit_map;

            
                if (bt_inners[offset].keys[pos] == bt_inners[offset].keys[pos + 1]){
                    //update key
                    bt_inners[offset].keys[pos] = keyNextLevel;

                    bit_map = bit_map ^ (0x8000 >> pos);

                    if (bt_inners[offset].keys[pos + 1] != ULONG_MAX)
                    {
                        bt_inners[offset].ptrs[pos + 1] = ptrNextLevel;
                    }
                    else
                    {
                        for (uint32_t p = pos + 1; p < BLOCKSIZE; p++)
                        {
                            bt_inners[offset].ptrs[p] = ptrNextLevel;
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
                        bt_inners[offset].keys[pos-1] = keyNextLevel;

                        //update pointers
                        memcpy(&bt_inners[offset].ptrs[start_copy], &bt_inners[offset].ptrs[start_copy + 1] , shift_pos* sizeof(uint32_t) );
                        bt_inners[offset].ptrs[pos] = ptrNextLevel;
                    }
                    else{

                        //update keys
                        uint16_t bit_flip = 0x8000;
                        uint32_t diff = shift_pos;// - 16;
                        memcpy(&bt_inners[offset].keys[pos + 1], &bt_inners[offset].keys[pos] , diff * sizeof(uint64_t) );
                        bit_map = bit_map ^ (bit_flip >> (pos + diff));
                        bt_inners[offset].keys[pos] = keyNextLevel;
                        
                        memcpy(&bt_inners[offset].ptrs[pos + 1], &bt_inners[offset].ptrs[pos] , diff  * sizeof(uint32_t) );
                        bt_inners[offset].ptrs[pos+1] = ptrNextLevel;
                    }
                }

                inners_stats[offset].slot_use++;
                inners_stats[offset].bit_map = ~bit_map;
                
                bt_inners[offset].writeUnlock();
                return;
            }
            else
            {
                bt_inners[parent_offset].upgradeToWriteLockOrRestart(versionParent, needRestart);
                if (needRestart)
                {
                    goto restartSplit;
                }
                
                
                bt_inners[offset].upgradeToWriteLockOrRestart(versionNode, needRestart);
                
                if (needRestart)
                {
                    bt_inners[parent_offset].writeUnlock();
                    goto restartSplit;
                }

                __builtin_prefetch(bt_inners[offset].ptrs);
                pos = successorLinG(bt_inners[offset].keys,keyNextLevel);

                uint32_t new_offset = bt_stats.num_inners.fetch_add(1);

                if (new_offset >= bt_stats.max_num_inners.load())
                {
                    reallocate_inners();
                }

                uint64_t temp_keys[BLOCKSIZE];
                uint32_t temp_ptrs[BLOCKSIZE + 1];
                mid_in_node = BLOCKSIZE/2;


                memcpy(&temp_keys, &bt_inners[offset].keys, pos * sizeof(uint64_t));
                temp_keys[pos] = keyNextLevel;

                memcpy(&temp_ptrs, &bt_inners[offset].ptrs, (pos + 1) * sizeof(uint32_t));
                temp_ptrs[pos + 1] = ptrNextLevel;

                if (keyNextLevel < bt_inners[offset].keys[BLOCKSIZE - 2])
                {
                    memcpy(&temp_keys[pos + 1], &bt_inners[offset].keys[pos], (BLOCKSIZE - pos - 1) * sizeof(uint64_t));
                    memcpy(&temp_ptrs[pos + 2], &bt_inners[offset].ptrs[pos + 1], (BLOCKSIZE - pos - 1) * sizeof(uint32_t));
                }

                bt_inners[offset].keys[0] = temp_keys[0];
                bt_inners[offset].ptrs[0] = temp_ptrs[0];

                uint32_t k = 1;
                for (i = 1; i < BLOCKSIZE - 1; i += 2, k++)
                {
                    bt_inners[offset].keys[i] = temp_keys[k];
                    bt_inners[offset].keys[i + 1] = temp_keys[k];

                    bt_inners[offset].ptrs[i] = temp_ptrs[k];
                    bt_inners[offset].ptrs[i + 1] = temp_ptrs[k];
                }

                bt_inners[offset].keys[BLOCKSIZE - 1] = ULONG_MAX;
                bt_inners[offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];

                keyNextLevel = temp_keys[k++];
                ptrNextLevel = new_offset;



                for (i = 0; i < BLOCKSIZE - 2; i += 2, k++)
                {
                    bt_inners[new_offset].keys[i] = temp_keys[k];
                    bt_inners[new_offset].keys[i + 1] = temp_keys[k];

                    bt_inners[new_offset].ptrs[i] = temp_ptrs[k];
                    bt_inners[new_offset].ptrs[i + 1] = temp_ptrs[k];
                }

                bt_inners[new_offset].keys[BLOCKSIZE - 2] = ULONG_MAX;
                bt_inners[new_offset].keys[BLOCKSIZE - 1] = ULONG_MAX;

                bt_inners[new_offset].ptrs[BLOCKSIZE - 2] = temp_ptrs[k];
                bt_inners[new_offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];

                //slotuse and bitmaps for previous node
                inners_stats[parent_offset].slot_use = mid_in_node;
                inners_stats[parent_offset].bit_map = 0xaaab;

                //slotuse and bitmaps for new node
                inners_stats[new_offset].slot_use = mid_in_node-1;
                inners_stats[new_offset].bit_map = 0x5555;

                bt_inners[offset].writeUnlock();


                __builtin_prefetch(bt_inners[parent_offset].ptrs);
                pos = successorLinG(bt_inners[parent_offset].keys,keyNextLevel);

                if (inners_stats[parent_offset].slot_use < BLOCKSIZE - 1)
                {
                    uint32_t shift_pos = 0;
                    uint16_t bit_map = ~inners_stats[parent_offset].bit_map;

                    if (bt_inners[parent_offset].keys[pos] == bt_inners[parent_offset].keys[pos + 1]){
                        //update key
                        bt_inners[parent_offset].keys[pos] = keyNextLevel;

                        bit_map = bit_map ^ (0x8000 >> pos);

                        if (bt_inners[parent_offset].keys[pos + 1] != ULONG_MAX)
                        {
                            bt_inners[parent_offset].ptrs[pos + 1] = ptrNextLevel;
                        }
                        else
                        {
                            for (uint32_t p = pos + 1; p < BLOCKSIZE; p++)
                            {
                                bt_inners[parent_offset].ptrs[p] = ptrNextLevel;
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
                            memcpy(&bt_inners[parent_offset].keys[start_copy], &bt_inners[parent_offset].keys[start_copy + 1] , (shift_pos-1) * sizeof(uint64_t) );
                            bit_map = bit_map ^ (bit_flip >> start_copy);
                            bt_inners[parent_offset].keys[pos-1] = keyNextLevel;

                            //update pointers
                            memcpy(&bt_inners[parent_offset].ptrs[start_copy], &bt_inners[parent_offset].ptrs[start_copy + 1] , shift_pos* sizeof(uint32_t) );
                            bt_inners[parent_offset].ptrs[pos] = ptrNextLevel;
                        }
                        else{

                            //update keys
                            uint16_t bit_flip = 0x8000;
                            uint32_t diff = shift_pos;// - 16;
                            memcpy(&bt_inners[parent_offset].keys[pos + 1], &bt_inners[parent_offset].keys[pos] , diff * sizeof(uint64_t) );
                            bit_map = bit_map ^ (bit_flip >> (pos + diff));
                            bt_inners[parent_offset].keys[pos] = keyNextLevel;
                            
                            memcpy(&bt_inners[parent_offset].ptrs[pos + 1], &bt_inners[parent_offset].ptrs[pos] , diff  * sizeof(uint32_t) );
                            bt_inners[parent_offset].ptrs[pos+1] = ptrNextLevel;
                        }
                    }

                        inners_stats[parent_offset].slot_use++;
                        inners_stats[parent_offset].bit_map = ~bit_map;

                        bt_inners[parent_offset].writeUnlock();

                        return;
                    }
                else    // split at inner node is needed
                {
                    uint32_t new_offset = bt_stats.num_inners.fetch_add(1);

                    if (new_offset >= bt_stats.max_num_inners.load())
                    {
                        reallocate_inners();
                    }

                    uint64_t temp_keys[BLOCKSIZE];
                    uint32_t temp_ptrs[BLOCKSIZE + 1];
                    mid_in_node = BLOCKSIZE/2;


                    memcpy(&temp_keys, &bt_inners[parent_offset].keys, pos * sizeof(uint64_t));
                    temp_keys[pos] = keyNextLevel;

                    memcpy(&temp_ptrs, &bt_inners[parent_offset].ptrs, (pos + 1) * sizeof(uint32_t));
                    temp_ptrs[pos + 1] = ptrNextLevel;

                    if (keyNextLevel < bt_inners[parent_offset].keys[BLOCKSIZE - 2])
                    {
                        memcpy(&temp_keys[pos + 1], &bt_inners[parent_offset].keys[pos], (BLOCKSIZE - pos - 1) * sizeof(uint64_t));
                        memcpy(&temp_ptrs[pos + 2], &bt_inners[parent_offset].ptrs[pos + 1], (BLOCKSIZE - pos - 1) * sizeof(uint32_t));
                    }

                    bt_inners[parent_offset].keys[0] = temp_keys[0];
                    bt_inners[parent_offset].ptrs[0] = temp_ptrs[0];
        
                    uint32_t k = 1;
                    for (i = 1; i < BLOCKSIZE - 1; i += 2, k++)
                    {
                        bt_inners[parent_offset].keys[i] = temp_keys[k];
                        bt_inners[parent_offset].keys[i + 1] = temp_keys[k];
        
                        bt_inners[parent_offset].ptrs[i] = temp_ptrs[k];
                        bt_inners[parent_offset].ptrs[i + 1] = temp_ptrs[k];
                    }

                    bt_inners[parent_offset].keys[BLOCKSIZE - 1] = ULONG_MAX;
                    bt_inners[parent_offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];
        
                    keyNextLevel = temp_keys[k++];
                    ptrNextLevel = new_offset;



                    for (i = 0; i < BLOCKSIZE - 2; i += 2, k++)
                    {
                        bt_inners[new_offset].keys[i] = temp_keys[k];
                        bt_inners[new_offset].keys[i + 1] = temp_keys[k];
        
                        bt_inners[new_offset].ptrs[i] = temp_ptrs[k];
                        bt_inners[new_offset].ptrs[i + 1] = temp_ptrs[k];
                    }
        
                    bt_inners[new_offset].keys[BLOCKSIZE - 2] = ULONG_MAX;
                    bt_inners[new_offset].keys[BLOCKSIZE - 1] = ULONG_MAX;
        
                    bt_inners[new_offset].ptrs[BLOCKSIZE - 2] = temp_ptrs[k];
                    bt_inners[new_offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];
        
                    //slotuse and bitmaps for previous node
                    inners_stats[parent_offset].slot_use = mid_in_node;
                    inners_stats[parent_offset].bit_map = 0xaaab;
        
                    //slotuse and bitmaps for new node
                    inners_stats[new_offset].slot_use = mid_in_node-1;
                    inners_stats[new_offset].bit_map = 0x5555;

                    bt_inners[parent_offset].writeUnlock();

                    currentDepth += 2;

                    goto restartSplit;
                }
            }
        }
        else
        {

            if (inners_stats[offset].slot_use < BLOCKSIZE -1)
            {

                bt_inners[offset].upgradeToWriteLockOrRestart(versionNode, needRestart);

                if (needRestart)
                {
                    goto restartSplit;
                }

                __builtin_prefetch(bt_inners[offset].ptrs);
                pos = successorLinG(bt_inners[offset].keys,keyNextLevel);


                uint32_t shift_pos = 0;
                uint16_t bit_map = ~inners_stats[offset].bit_map;

            
                if (bt_inners[offset].keys[pos] == bt_inners[offset].keys[pos + 1]){
                    //update key
                    bt_inners[offset].keys[pos] = keyNextLevel;

                    bit_map = bit_map ^ (0x8000 >> pos);

                    if (bt_inners[offset].keys[pos + 1] != ULONG_MAX)
                    {
                        bt_inners[offset].ptrs[pos + 1] = ptrNextLevel;
                    }
                    else
                    {
                        for (uint32_t p = pos + 1; p < BLOCKSIZE; p++)
                        {
                            bt_inners[offset].ptrs[p] = ptrNextLevel;
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
                        bt_inners[offset].keys[pos-1] = keyNextLevel;

                        //update pointers
                        memcpy(&bt_inners[offset].ptrs[start_copy], &bt_inners[offset].ptrs[start_copy + 1] , shift_pos* sizeof(uint32_t) );
                        bt_inners[offset].ptrs[pos] = ptrNextLevel;
                    }
                    else{

                        //update keys
                        uint16_t bit_flip = 0x8000;
                        uint32_t diff = shift_pos;// - 16;
                        memcpy(&bt_inners[offset].keys[pos + 1], &bt_inners[offset].keys[pos] , diff * sizeof(uint64_t) );
                        bit_map = bit_map ^ (bit_flip >> (pos + diff));
                        bt_inners[offset].keys[pos] = keyNextLevel;
                        
                        memcpy(&bt_inners[offset].ptrs[pos + 1], &bt_inners[offset].ptrs[pos] , diff  * sizeof(uint32_t) );
                        bt_inners[offset].ptrs[pos+1] = ptrNextLevel;
                    }
                }

                inners_stats[offset].slot_use++;
                inners_stats[offset].bit_map = ~bit_map;
                
                bt_inners[offset].writeUnlock();
                return;

            }
            else
            {
                bt_inners[offset].upgradeToWriteLockOrRestart(versionNode, needRestart);

                if (needRestart)
                {
                    goto restartSplit;
                }

                __builtin_prefetch(bt_inners[offset].ptrs);
                pos = successorLinG(bt_inners[offset].keys,keyNextLevel);

                uint32_t new_offset = bt_stats.num_inners.fetch_add(1);

                if (new_offset >= bt_stats.max_num_inners.load())
                {
                    reallocate_inners();
                }

                uint64_t temp_keys[BLOCKSIZE];
                uint32_t temp_ptrs[BLOCKSIZE + 1];
                mid_in_node = BLOCKSIZE/2;


                memcpy(&temp_keys, &bt_inners[offset].keys, pos * sizeof(uint64_t));
                temp_keys[pos] = keyNextLevel;

                memcpy(&temp_ptrs, &bt_inners[offset].ptrs, (pos + 1) * sizeof(uint32_t));
                temp_ptrs[pos + 1] = ptrNextLevel;

                if (keyNextLevel < bt_inners[offset].keys[BLOCKSIZE - 2])
                {
                    memcpy(&temp_keys[pos + 1], &bt_inners[offset].keys[pos], (BLOCKSIZE - pos - 1) * sizeof(uint64_t));
                    memcpy(&temp_ptrs[pos + 2], &bt_inners[offset].ptrs[pos + 1], (BLOCKSIZE - pos - 1) * sizeof(uint32_t));
                }

                bt_inners[offset].keys[0] = temp_keys[0];
                bt_inners[offset].ptrs[0] = temp_ptrs[0];

                uint32_t k = 1;
                for (i = 1; i < BLOCKSIZE - 1; i += 2, k++)
                {
                    bt_inners[offset].keys[i] = temp_keys[k];
                    bt_inners[offset].keys[i + 1] = temp_keys[k];

                    bt_inners[offset].ptrs[i] = temp_ptrs[k];
                    bt_inners[offset].ptrs[i + 1] = temp_ptrs[k];
                }

                bt_inners[offset].keys[BLOCKSIZE - 1] = ULONG_MAX;
                bt_inners[offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];

                keyNextLevel = temp_keys[k++];
                ptrNextLevel = new_offset;

                for (i = 0; i < BLOCKSIZE - 2; i += 2, k++)
                {
                    bt_inners[new_offset].keys[i] = temp_keys[k];
                    bt_inners[new_offset].keys[i + 1] = temp_keys[k];

                    bt_inners[new_offset].ptrs[i] = temp_ptrs[k];
                    bt_inners[new_offset].ptrs[i + 1] = temp_ptrs[k];
                }

                bt_inners[new_offset].keys[BLOCKSIZE - 2] = ULONG_MAX;
                bt_inners[new_offset].keys[BLOCKSIZE - 1] = ULONG_MAX;

                bt_inners[new_offset].ptrs[BLOCKSIZE - 2] = temp_ptrs[k];
                bt_inners[new_offset].ptrs[BLOCKSIZE - 1] = temp_ptrs[k];

                //slotuse and bitmaps for previous node
                inners_stats[parent_offset].slot_use = mid_in_node;
                inners_stats[parent_offset].bit_map = 0xaaab;

                //slotuse and bitmaps for new node
                inners_stats[new_offset].slot_use = mid_in_node-1;
                inners_stats[new_offset].bit_map = 0x5555;

                uint32_t new_root_offset = bt_stats.num_inners.fetch_add(1);

                bt_inners[new_root_offset].keys[0] = keyNextLevel;
                bt_inners[new_root_offset].ptrs[0] = offset;

                for (uint32_t p = 1; p < BLOCKSIZE; p++)
                {
                    bt_inners[new_root_offset].keys[p] = ULONG_MAX;
                    bt_inners[new_root_offset].ptrs[p] = ptrNextLevel;
                }

                inners_stats[new_root_offset].slot_use = 1;
                inners_stats[new_root_offset].bit_map = 0x8001;

                uint32_t old_root_offset = root_offset.load();
                root_offset.compare_exchange_strong(old_root_offset, new_root_offset);

                uint32_t oldHeight = bt_stats.height.load();
                bt_stats.height.compare_exchange_weak(oldHeight, oldHeight + 1);
                
                bt_inners[offset].writeUnlock();

                return;

            }
        }
}
#endif