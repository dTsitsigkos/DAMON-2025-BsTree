#ifndef _BSTREE_H_
#define _BSTREE_H_

#include <bits/stdc++.h>
#include <immintrin.h>
#include "def.h"

//global variables about the leaves
btree_leaf *bt_leaves;

//print leaves nodes
void print_leaves(){

    for (uint32_t i = 0; i < bt_stats.num_leaves; i++){   
        
        printf("For Leaf Node %u: \n", i);
        printf("- Keys:\n");
        for (uint32_t j = 0; j < BLOCKSIZE_64BIT; j++){
            printf("%lu ", bt_leaves[i].keys[j]);
        }
        printf("\n- Statistics:\n");
        printf("Slotuse = %d -- Bitmap = 0x%X -- nextLeaf = %u\n\n", leaves_stats[i].slot_use, leaves_stats[i].bit_map, leaves_stats[i].next_leaf);
    }
}


//print inner nodes
void print_inners(){
    
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
void reallocate_leaves(){
    btree_leaf *old_ptr = bt_leaves;
    btree_leaf *new_ptr;
    stats_leaf *old_stats = leaves_stats, *new_stats;

    //increase the number of leaves
    bt_stats.max_num_leaves = 2 * bt_stats.num_leaves;

    //allocate new array for leaves
    new_ptr = (btree_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(btree_leaf));
    
    if (new_ptr == NULL){
        printf("BAD ALLOCATION AT LEAF NODES\n");
        exit(-1);
    }

    //allocate new array for leaves stats
    new_stats = (stats_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(stats_leaf));

    if (new_stats == NULL){
        printf("BAD ALLOCATION AT LEAF STATS\n");
        exit(-1);
    }

    //copy the previous array of leaves to the new ones
    memcpy(new_ptr, old_ptr, bt_stats.num_leaves * sizeof(btree_leaf));
    bt_leaves = new_ptr;

    //copy the previous array of leaves stats to the new ones
    memcpy(new_stats, old_stats, bt_stats.num_leaves * sizeof(stats_leaf));
    leaves_stats = new_stats;

    //free the previous arrays
    free(old_ptr);
    free(old_stats);
}


// reallocation in inners
void reallocate_inners(){
    btree_inner *old_ptr = bt_inners;
    btree_inner *new_ptr;
    stats_inner *old_stats = inners_stats, *new_stats;    

    //increase the number of inners
    bt_stats.max_num_inners =  2* bt_stats.max_num_inners; //initialize max num inner nodes

    //allocate new array for innert using huge pages
    uint32_t num_pages_inner = ((bt_stats.max_num_inners * sizeof(btree_inner) -1) / page_size) + 1;//initialize num pages for inner nodes
    new_ptr = (btree_inner *) aligned_alloc(page_size, num_pages_inner * page_size);
    madvise(bt_inners, num_pages_inner*page_size, MADV_HUGEPAGE);

    if (new_ptr == NULL){
        printf("BAD ALLOCATION AT INNER NODES\n");
        exit(-1);
    }

    //allocate new array for inner stats
    new_stats = (stats_inner *) aligned_alloc(64, bt_stats.max_num_inners*sizeof(stats_inner));
    
    if (new_stats == NULL){
        printf("BAD ALLOCATION AT INNER STATS\n");
        exit(-1);
    }

    //copy the previous array of inners to the new ones
    memcpy(new_ptr, old_ptr, bt_stats.num_inners * sizeof(btree_inner));
    bt_inners = new_ptr;

    //copy the previous array of inners stats to the new ones
    memcpy(new_stats, old_stats, bt_stats.num_inners * sizeof(stats_inner));
    inners_stats = new_stats;

    //free the previous arrays
    free(old_ptr);
    free(old_stats);
}


// finds lowest position greater than key with count
inline uint32_t successorLinG(uint64_t *ar, uint64_t key){
    uint32_t s1 = 0;

    __m512i threshold_vector = _mm512_set1_epi64(key); //load the search key into one register

    __m512i y1 = _mm512_loadu_epi64((__m512i*) ar); // load the first 8 keys from the node into one register (y1)
    uint32_t m1 = (uint32_t) _mm512_cmpge_epu64_mask(threshold_vector,y1);//compare threshold_vector >= y1
    s1 = _mm_popcnt_u32(m1); //pop count the result of the comparison

    __m512i y2 = _mm512_loadu_epi64((__m512i*)(ar + 8)); // load the last 8 keys from the node into one register (y2)
    uint32_t m2 = (uint32_t) _mm512_cmpge_epu64_mask(threshold_vector,y2);//compare threshold_vector >= y2
    s1 += _mm_popcnt_u32(m2); //pop count the result of the comparison and add it with the previous one

    return s1;
}


// finds lowest position greater than or equal to key with count
inline uint32_t successorLinGE(uint64_t *ar, uint64_t key){
    uint32_t s1 = 0;

    __m512i threshold_vector = _mm512_set1_epi64(key); //load the search key into one register
	
    __m512i y1 = _mm512_loadu_epi64((__m512i*) ar); // load the first 8 keys from the node into one register (y1)
    uint32_t m1 = (uint32_t) _mm512_cmpgt_epu64_mask(threshold_vector, y1); //compare threshold_vector > y1
    s1 = _mm_popcnt_u32(m1); //pop count the result of the comparison

    __m512i y2 = _mm512_loadu_epi64((__m512i*) (ar + 8)); // load the last 8 keys from the node into one register (y2)
    uint32_t m2 = (uint32_t) _mm512_cmpgt_epu64_mask(threshold_vector, y2); //compare threshold_vector > y2
    s1 += _mm_popcnt_u32(m2); //pop count the result of the comparison and add it with the previous one
    
    s1 = (s1 == BLOCKSIZE_64BIT) ? --s1 : s1;

    return s1;
}


//construct the leaves nodes of the bstree
uint64_t *construct_leaves(uint64_t *ar, uint64_t size_data){   
    uint32_t i = 0, j = 0;
    uint64_t limited_size = 3 << 30;    //limit of transparent huge pages
    uint64_t NUMnextlevelnonleafkeys = 0, prev_inserted_leaf = 0;    
    uint64_t * nextlevelnonleafkeys;

    //allocate leaves nodes and leaves statistics array
    bt_stats.num_leaves = ceil((double)size_data / KEYS_IN_LEAF_64BIT);   //initialize num leaves based on the num of gaps   
    bt_stats.max_num_leaves = 2 * bt_stats.num_leaves;

    if (bt_stats.max_num_leaves * sizeof(btree_leaf) <= limited_size){// leaves nodes into huge pages
        uint32_t num_pages_leaves = ((bt_stats.max_num_leaves * sizeof(btree_leaf) -1) / page_size) + 1;
        bt_leaves = (btree_leaf *) aligned_alloc(page_size,  num_pages_leaves*page_size);//allocation memory for leaves
        madvise(bt_leaves, num_pages_leaves * page_size, MADV_HUGEPAGE);
    }
    else{//leaves nodes not into huge pages
        bt_leaves = (btree_leaf *) aligned_alloc(64, bt_stats.max_num_leaves * sizeof(btree_leaf));
    }
    leaves_stats = (stats_leaf *) aligned_alloc(64,bt_stats.max_num_leaves*sizeof(stats_leaf));   // allocate the additional space for the statistics of the leaves
 
    //allocate array for the keys (separators) stored in non-leaf nodes (next level)
    nextlevelnonleafkeys = (uint64_t *)aligned_alloc(64, sizeof(uint64_t) * bt_stats.num_leaves);

    //create all node leaves, except the last one
    for (i = 0; prev_inserted_leaf + KEYS_IN_LEAF_64BIT < size_data; i++){
        uint64_t diff = ar[prev_inserted_leaf + BLOCKSIZE_64BIT -1] - ar[prev_inserted_leaf];

        //sequential keys
        if (diff == BLOCKSIZE_64BIT - 1){
            leaves_stats[i].slot_use = BLOCKSIZE_64BIT;
            leaves_stats[i].bit_map = 0xffff;

            memcpy(bt_leaves[i].keys, &ar[prev_inserted_leaf], BLOCKSIZE_64BIT * sizeof(uint64_t));
            prev_inserted_leaf += BLOCKSIZE_64BIT;
         
        }
        else{//non-sequential keys, three keys one gap
            leaves_stats[i].slot_use = KEYS_IN_LEAF_64BIT;
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

            bt_leaves[i].keys[BLOCKSIZE_64BIT - 1] = ULONG_MAX;
        }

       leaves_stats[i].next_leaf = i+1;
        if (i>0) nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = bt_leaves[i].keys[0];
    }

    // last leaf
    int residue = (size_data - prev_inserted_leaf);
   
    uint16_t temp_bit_map = 0xffff;
    temp_bit_map = temp_bit_map << (BLOCKSIZE_64BIT - residue);
    
    leaves_stats[i].slot_use = residue;
    leaves_stats[i].bit_map = temp_bit_map;
                
    memcpy(&bt_leaves[i].keys,&ar[prev_inserted_leaf],residue*sizeof(uint64_t));
    for (j = residue;j<BLOCKSIZE_64BIT ;j++){
        bt_leaves[i].keys[j] = ULONG_MAX;
        
    }
    nextlevelnonleafkeys[NUMnextlevelnonleafkeys++] = bt_leaves[i].keys[0];
    leaves_stats[i].next_leaf = 0;
    
    bt_stats.num_leaves = i + 1;

    return nextlevelnonleafkeys;
}


//construct the inner nodes of the bstree
void construct_inners(uint64_t *nextlevelnonleafkeys){
    uint64_t i = 0, j = 0;
    uint64_t NUMnextlevelnonleafkeys = bt_stats.num_leaves;
    uint64_t prev_inserted_inner = 0;
    uint32_t nextpointedblockid = 0; 
    uint32_t previous_level = 0;
    uint32_t numcblocks =  0; 
    uint32_t idx = 0;
    uint32_t levels_without_gaps = 1;

    //height optimized mechanism
    uint32_t height_16 = ceil(log2(bt_stats.num_leaves)/log2(BLOCKSIZE_64BIT));
    uint32_t height_15 = ceil(log2(bt_stats.num_leaves)/log2(KEYS_IN_INNER + 1));
    if (height_16 == height_15){
        bt_stats.height = height_15;
    }
    else{
        bt_stats.height = height_16;
        levels_without_gaps = floor(height_15/2);
    }

    //create and set values to the lut (lookup table)
    uint32_t lut[bt_stats.height]; 
    for (i = 0; i < bt_stats.height-levels_without_gaps; i++){// put gaps to the specific level (based on optimized mechanism)
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

        //create inner nodes with gaps (base on the height optimized mechanism)
        for (; level < bt_stats.height-levels_without_gaps; level ++){
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

                    if ( t < BLOCKSIZE_64BIT -1 ){
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
bool search(uint64_t skey){
    uint32_t i, pos, offset = root_offset;
    
    for (i = bt_stats.height; i > 0; i--){
        __builtin_prefetch(bt_inners[offset].ptrs);//prefetch the pointers to the next node
	    pos = successorLinG(bt_inners[offset].keys,skey); // search in inner nodes
	    offset = bt_inners[offset].ptrs[pos];
	}
	pos = successorLinGE(bt_leaves[offset].keys,skey); //search in leaves
	
    if (bt_leaves[offset].keys[pos]==skey)
        return 1;
    else
	    return 0;
}


//insert function in the bstree
//General idea is that for a gap, at first we are looking at the right of the position and then at the right
//if a leaf node, does not hava space, we split the node, and we insert the gaps proactively.
//Same functionality with the inners.
void insert_key(uint64_t ikey){
    uint32_t i, pos, offset = root_offset;
    uint32_t tree_path[bt_stats.height];

    //search to find the position of the key, where the key should be inserted
    for (i = bt_stats.height; i > 0; i--){
        __builtin_prefetch(bt_inners[offset].ptrs);
	    pos = successorLinG(bt_inners[offset].keys,ikey);
        tree_path[i-1] = offset;
        offset = bt_inners[offset].ptrs[pos];
    }

    pos = successorLinGE(bt_leaves[offset].keys,ikey);

    //if the key is already exists, then we return
    if (bt_leaves[offset].keys[pos] == ikey){
        return;
    }

    uint8_t slot_use = leaves_stats[offset].slot_use;

    //if the leaf has space
    if (slot_use < BLOCKSIZE_64BIT) {// we have empty space
        
        uint16_t bit_map = leaves_stats[offset].bit_map;
        bit_map = ~bit_map;
 
        if (slot_use != 0){//if the leaf in not empty
            uint32_t shift_pos = 0;

            if (pos == 0){// if key has to be inserted in the first position of the node, we try to find a gap in the right of the position

                shift_pos = _lzcnt_u32((uint32_t)bit_map);
                
                if (shift_pos > 16){
                    uint16_t bit_flip = 0x8000;
                    uint32_t diff = shift_pos - 16;

                    memcpy(&bt_leaves[offset].keys[pos + 1], &bt_leaves[offset].keys[pos] , diff * sizeof(uint64_t) );
                    bit_map = bit_map ^ (bit_flip >> diff);
                }
                else{
                    bit_map = bit_map ^ 0x8000;
                }
                bt_leaves[offset].keys[pos] = ikey;           
            }
            else if (pos == BLOCKSIZE_64BIT - 1){//if key has to be inserted in the last position of the node, we try to find a gap in the left of the position

                shift_pos = _tzcnt_u16(bit_map);
            
                if (shift_pos > 0){
                    uint16_t bit_flip = 0x0001;
                    uint32_t start_copy = pos - shift_pos;
                    uint32_t keysForCopy = shift_pos;

                    if (ikey < bt_leaves[offset].keys[BLOCKSIZE_64BIT - 1]){
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
            else{ //if key has to be inserted in any other position of the node, at first we try to find a gap in right of the position and then to the left

                if (bt_leaves[offset].keys[pos] == bt_leaves[offset].keys[pos + 1]){
                    bt_leaves[offset].keys[pos] = ikey;
                    bit_map = bit_map ^ (0x8000 >> pos);

                }
                else{
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
        else{//if the leaf is completely empty
            bt_leaves[offset].keys[0] = ikey;
            bit_map = bit_map ^ 0x8000;
        }
        
        //update the stats of the leaf
        leaves_stats[offset].slot_use++;        
        leaves_stats[offset].bit_map = ~bit_map;

        return;
    }


    //if the leaf node does not has space, then we should split the node
    //first check if we have space for a new leaf   
    if (bt_stats.num_leaves == bt_stats.max_num_leaves){
        reallocate_leaves();
    }

    uint64_t temp_arr[BLOCKSIZE_64BIT+1];
    uint32_t mid_in_node = BLOCKSIZE_64BIT/2;
    uint64_t key_for_next_level = 0;
    uint32_t ptr_for_next_level = 0;

    //create a temp array for the leaf to help us for the spliting 
    memcpy(&temp_arr, &bt_leaves[offset].keys, pos * sizeof(uint64_t));
    if(ikey > bt_leaves[offset].keys[BLOCKSIZE_64BIT - 1]){
        temp_arr[pos] = bt_leaves[offset].keys[BLOCKSIZE_64BIT - 1];
        temp_arr[pos + 1] = ikey;
    }
    else{
        temp_arr[pos] = ikey;
        memcpy(&temp_arr[pos+1], &bt_leaves[offset].keys[pos], (BLOCKSIZE_64BIT - pos)*sizeof(uint64_t));
    }

    if (pos > mid_in_node){//new key in the right child

        //initialize the first key in the previous and new node
        bt_leaves[offset].keys[0] = temp_arr[0];
        bt_leaves[bt_stats.num_leaves].keys[0] = temp_arr[mid_in_node];

        //insert keys and gaps in the previous and new node without the last key
        uint32_t j = 1;
        for (uint32_t i = 1 ; i < BLOCKSIZE_64BIT -1  ; i += 2){
            bt_leaves[offset].keys[i] = temp_arr[j];
            bt_leaves[offset].keys[i+1] = temp_arr[j];

            bt_leaves[bt_stats.num_leaves].keys[i] = temp_arr[mid_in_node + j];
            bt_leaves[bt_stats.num_leaves].keys[i+1] = temp_arr[mid_in_node + j];
            j++;
        }

        //insert last key in the previous and new node
        bt_leaves[offset].keys[BLOCKSIZE_64BIT -1] = ULONG_MAX;
        bt_leaves[bt_stats.num_leaves].keys[BLOCKSIZE_64BIT-1] = temp_arr[BLOCKSIZE_64BIT];

        //create stats for new node
        leaves_stats[bt_stats.num_leaves].slot_use = mid_in_node + 1;
        leaves_stats[bt_stats.num_leaves].bit_map = 0xaaab;
        
        //update stats for previous node
        leaves_stats[offset].slot_use = mid_in_node;
        leaves_stats[offset].bit_map = 0xaaaa;
    }
    else{ //new key in left child

        //initialize the first key in the previous and new node
        bt_leaves[offset].keys[0] = temp_arr[0];
        bt_leaves[bt_stats.num_leaves].keys[0] = temp_arr[mid_in_node + 1];

        //insert keys and gaps in the previous and new node without the last key
        uint32_t j = 1;
        for (uint32_t i = 1 ; i < BLOCKSIZE_64BIT -1 ; i += 2){
            bt_leaves[offset].keys[i] = temp_arr[j];
            bt_leaves[offset].keys[i+1] = temp_arr[j];

            bt_leaves[bt_stats.num_leaves].keys[i] = temp_arr[mid_in_node + 1 + j];
            bt_leaves[bt_stats.num_leaves].keys[i+1] = temp_arr[mid_in_node + 1 + j];
            j++;
        }

        //insert last key in the previous and new node   
        bt_leaves[offset].keys[BLOCKSIZE_64BIT-1] = temp_arr[mid_in_node];
        bt_leaves[bt_stats.num_leaves].keys[BLOCKSIZE_64BIT -1] = ULONG_MAX;
 
        //update stats for previous node        
        leaves_stats[offset].slot_use = mid_in_node + 1;
        leaves_stats[offset].bit_map = 0xaaab;

        //update new node
        leaves_stats[bt_stats.num_leaves].slot_use = mid_in_node;
        leaves_stats[bt_stats.num_leaves].bit_map = 0xaaaa;
    }

    //update next leaf pointer for new node
    leaves_stats[bt_stats.num_leaves].next_leaf = leaves_stats[offset].next_leaf; 

    //update next leaf pointer for previous node
    leaves_stats[offset].next_leaf = bt_stats.num_leaves;
    
    //get the key for next level
    key_for_next_level = bt_leaves[bt_stats.num_leaves].keys[0];
    
    //update ptr for the next level
    ptr_for_next_level = bt_stats.num_leaves;
    bt_stats.num_leaves += 1;

    //insert in the inners above
    for (uint32_t j = 0 ; j < bt_stats.height; j++){
        
        offset = tree_path[j];
        __builtin_prefetch(bt_inners[offset].ptrs);
        pos = successorLinG(bt_inners[offset].keys,key_for_next_level);

        slot_use = inners_stats[offset].slot_use;

        if (slot_use < BLOCKSIZE_64BIT -1){//we have empty slot
            uint32_t shift_pos = 0;
            uint16_t bit_map = inners_stats[offset].bit_map;
            bit_map = ~bit_map;
        
        
            if (bt_inners[offset].keys[pos] == bt_inners[offset].keys[pos + 1]){ //if pos == gap 
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
            else{// fing the next gap, first at the right of the pos, then at the left
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
                    uint32_t diff = shift_pos;// - 16;
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

            //if we need reallocation in the inners
            if (bt_stats.num_inners == bt_stats.max_num_inners){
                reallocate_inners();
            }
            
            mid_in_node = BLOCKSIZE_64BIT/2;

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
                    if (pos == mid_in_node - 1)
                    {
                        bt_inners[offset].keys[BLOCKSIZE_64BIT - 1] = ULONG_MAX;
                        bt_inners[offset].ptrs[BLOCKSIZE_64BIT - 1] = ptr_for_next_level;

                        bt_inners[offset].keys[BLOCKSIZE_64BIT - 2] = inserted_in_pos;
                        bt_inners[offset].keys[BLOCKSIZE_64BIT - 3] = inserted_in_pos;
                        
                        bt_inners[offset].ptrs[BLOCKSIZE_64BIT - 2] = bt_inners[offset].ptrs[mid_in_node - 1];
                        bt_inners[offset].ptrs[BLOCKSIZE_64BIT - 3] = bt_inners[offset].ptrs[mid_in_node - 1];

                        k = mid_in_node - 2;

                        for (i = BLOCKSIZE_64BIT - 4; i > 0; i -= 2)
                        {
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

                        for (i = BLOCKSIZE_64BIT - 2; i > 2 * (pos + 1); i-= 2)
                        {
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

                        for (i = i -2; i > 0; i-= 2)
                        {

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
            inners_stats[bt_stats.num_inners].slot_use = mid_in_node-1;
            inners_stats[bt_stats.num_inners].bit_map = 0x5555;

            bt_stats.num_inners ++;

            //check if the split reaches the root
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


//delete key from the bstree
void delete_key(uint64_t dkey){
    uint32_t i, pos, offset = root_offset;
    uint8_t slot_use;
    uint16_t bit_map;
    uint32_t valid_pos;

    //search to find the position of the key, that we try to delete
    for (i = bt_stats.height; i > 0; i--){
        __builtin_prefetch(bt_inners[offset].ptrs);
	    pos = successorLinG(bt_inners[offset].keys,dkey);
	    offset = bt_inners[offset].ptrs[pos];
    }
    pos = successorLinGE(bt_leaves[offset].keys,dkey);

    //if key doesnot exist
    if (bt_leaves[offset].keys[pos] != dkey){
        return;
    }

    bit_map = leaves_stats[offset].bit_map;
    
    // key is not in the last position of the node
    if (pos != BLOCKSIZE_64BIT - 1){
        uint32_t temp_bit_map = bit_map << (pos + 16);
        valid_pos = _lzcnt_u32(temp_bit_map); //find the valid key (no gaps)
        
        uint64_t reproducedKey = bt_leaves[offset].keys[pos + 1 + valid_pos]; // the key after the valid key is the new new key, that has to be reproduced
        for (i = 0; i <= valid_pos; i++){// reproduce the reproducedKey from the pos, until the next valid pos
            bt_leaves[offset].keys[pos + i] = reproducedKey;
        }

        bit_map = bit_map ^ (0x8000 >> (pos + valid_pos));
        
    }
    else// key is in the last position of the node
    {
        bit_map = bit_map ^ 0x0001;
        valid_pos = _tzcnt_u16(bit_map); // find the previous valid position(not gap)

        for (i = 0; i < valid_pos; i++){ //reproduce the ULONG_ΜΑΧ
            bt_leaves[offset].keys[pos - i] = ULONG_MAX;
        }
    }

    leaves_stats[offset].slot_use--;
    leaves_stats[offset].bit_map = bit_map;
}


//range query for bstree
uint64_t range_search(uint64_t start_key, uint64_t end_key){
    int64_t count = 0;
    uint32_t start_range_offset = 0;
    uint32_t end_range_offset = 0;
    uint64_t upper_end_key = end_key + 1;
    uint32_t curr_offset = root_offset;
    uint32_t i, pos_start, pos_end;
    uint16_t bit_flip_16bit = 0x8000;

    //check if the query is wrong
    if(end_key <= start_key)
        return 0;

    // search to find the position of the start_key
    for (i = bt_stats.height; i > 0; i--){
        __builtin_prefetch(bt_inners[curr_offset].ptrs);
	    pos_start = successorLinG(bt_inners[curr_offset].keys,start_key);
	    curr_offset = bt_inners[curr_offset].ptrs[pos_start];
    }
    pos_start = successorLinGE(bt_leaves[curr_offset].keys,start_key);
    start_range_offset = curr_offset;
    
    //search to find the position of the end_key
    curr_offset = root_offset;
    for (i = bt_stats.height; i > 0; i--){
        __builtin_prefetch(bt_inners[curr_offset].ptrs);
	    pos_end = successorLinG(bt_inners[curr_offset].keys, upper_end_key);
	    curr_offset = bt_inners[curr_offset].ptrs[pos_end];
    }
    pos_end = successorLinGE(bt_leaves[curr_offset].keys, upper_end_key);
    end_range_offset = curr_offset;

    //update the position of pos_end if the upper_end_key is bigger number in the node
    pos_end = (upper_end_key > bt_leaves[end_range_offset].keys[BLOCKSIZE_64BIT - 1]) ? pos_end + 1 : pos_end;

    uint16_t bitmap = leaves_stats[start_range_offset].bit_map;
    bitmap = bitmap << pos_start;
    
    //start the process of the range query, first node of range
    if (start_range_offset == end_range_offset){// if both, start and end key are in the same node
        if (start_key > bt_leaves[start_range_offset].keys[BLOCKSIZE_64BIT-1])
            return 0;

        if(pos_start == pos_end){
            return 0;
        }
        
        for (i = pos_start; i < pos_end; i++){
            count += ((bitmap & bit_flip_16bit) >> (BLOCKSIZE_64BIT - 1));
            bitmap = bitmap << 1;
        }        

        return count;   
    }
    else{// start and end key are in different nodes

        // if start and end keys belonged to adjacent nodes
        if (start_key > bt_leaves[start_range_offset].keys[BLOCKSIZE_64BIT-1]){
            pos_start = 0;
            start_range_offset = leaves_stats[start_range_offset].next_leaf;
            bitmap = leaves_stats[start_range_offset].bit_map;

            if (start_range_offset == end_range_offset){
                for (i = pos_start; i < pos_end; i++){
                    count += ((bitmap & bit_flip_16bit) >> (BLOCKSIZE_64BIT - 1));
                    bitmap = bitmap << 1;
                }        

                return count;
            }    
        }
        
        // if start and end keys are not belonged to adjacent nodes
        for (i = pos_start; i < BLOCKSIZE_64BIT; i++){
            count += ((bitmap & bit_flip_16bit) >> (BLOCKSIZE_64BIT - 1));
            bitmap = bitmap << 1;
        }
    }

    curr_offset = leaves_stats[start_range_offset].next_leaf;

    //calculate the range query before the last leaf node of the range
    while (curr_offset != end_range_offset){          
        bitmap = leaves_stats[curr_offset].bit_map;

        for (i = 0 ; i < BLOCKSIZE_64BIT ; i++){
            count += ((bitmap & bit_flip_16bit) >> (BLOCKSIZE_64BIT - 1));
            bitmap = bitmap << 1;
        }

        curr_offset = leaves_stats[curr_offset].next_leaf;
    }

    bitmap = leaves_stats[end_range_offset].bit_map;

    //calculate the range query for the last leaf node of the range
    for (i = 0 ; i < pos_end ; i++){
        count += ((bitmap & bit_flip_16bit) >> (BLOCKSIZE_64BIT - 1));
        bitmap = bitmap << 1;
    }
    
    return count;
}


//delete the bstree
void delete_tree(){
    free(bt_leaves);
    free(bt_inners);
    cout<<"tree deleted"<<endl;
}

#endif