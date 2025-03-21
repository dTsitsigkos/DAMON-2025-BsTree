#ifndef _WORKLOADS_H_
#define _WORKLOADS_H_

#include "defOLC.h"
#include <sstream>
#include <cstdint>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 

#include "bstreeOLC.h"

//reads the construction dataset
inline uint64_t *read_dataset(char* input_data, uint64_t &size_data){
    uint64_t *ar;
    std::ifstream infile(input_data);

    if (infile.is_open()) { 
	    uint64_t num;
		
	    infile >> num;
	    size_data = num;

	    ar = (uint64_t*)malloc(sizeof(uint64_t) * size_data);
		
	    ulong i = 0;
	    while (infile >> num ){
		    ar[i] = num;
		    i++;
	    }

	    infile.close();
    }
    else{
        cerr << "Unable to open input file!" << endl; 
    }

    return ar;
}


//read the workloads
inline vector<std::array<uint64_t, 3>> read_work_load(char* input_data, uint64_t &size_data, int workload ){
    vector<std::array<uint64_t, 3>> workloads_arr ;
    std::ifstream infile(input_data);

    if (infile.is_open()) { 
        uint64_t num;
        int32_t w_id;
		
        infile >> num;  
        size_data = num;

        workloads_arr.resize(size_data);

        std::string line;
        uint64_t i = 0;
        
        getline(infile, line);
        while (getline(infile, line)) {         
            uint64_t j = 0;
            if (workload != -1){
                workloads_arr[i][j] = workload;
                j++;
            }
            
            if (line.find(' ') != std::string::npos){
                std::istringstream iss(line);  
                std::string token;
        
                while (std::getline(iss, token, ' ')) {
                    std::istringstream iss(token);
                    workloads_arr[i][j] = std::stoull(token);
                    j++;
                }
            }
            else{
                workloads_arr[i][1] = std::stoull(line);
            }         

            i++;
        }
    
        infile.close();
    }
    else{
        cerr << "Unable to open input file!" << endl; 
    }

    return workloads_arr;
}


//query executor bstree
inline bool execute_query(std::array<uint64_t, 3> q){
    uint64_t res = 0;
    switch(q[0]){
        case 0:
            // search
            res = search(q[1]);
            break;
        case 1:
            // insert
            insert(q[1]);
            break;
        case 2:
            //delete
            //delete_key(q[1]);
            break;
        case 3:
            //range
            //res = range_search(q[1], q[2]);
            break;
    }

    return res;
}


void memory_footprint(uint64_t size_data){
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        uint64_t size_ar = (size_data * sizeof(uint64_t) / 1024);
        std::cout << "Maximum Resident Set Size: " << usage.ru_maxrss - size_ar<< " KB" << std::endl;
        // Other fields can be printed similarly if needed
    } 
    else {
        std::cerr << "Error in getrusage" << std::endl;
    }
}


#endif