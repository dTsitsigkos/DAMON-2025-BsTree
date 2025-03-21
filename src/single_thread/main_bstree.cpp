#include "def.h"
#include "help_functions.h"
#include "bstree.h"


int main(int argc, char **argv){
    uint64_t i,j;
    uint64_t size_data = 0;
    uint64_t *ar; // auxiliary array for the initial data
    uint64_t *nextlevelnonleafkeys; // keys (separators) stored in non-leaf nodes (next level)
    vector<std::array<uint64_t, 3>> workloads_arr;//workloads are saved in this array

    //take the arguments
    char* construct_data = argv[1]; //path for the file with the construction data
    char* workload_data = argv[2];  //path for the file with the workloads
    printf("\nConstruction Dataset: %s\n", construct_data);
    printf("Workload Dataset: %s\n\n", workload_data);
    
    //read sort the construction data
    ar = read_dataset(construct_data, size_data);   //read dataset and saved to the auxiliary array
    sort(ar, ar + size_data); // sort the date in the auxiliary array
    
    //construction of the bstree
    auto constructionTime = 0;
    auto constructionStart =  high_resolution_clock::now();
    
    nextlevelnonleafkeys = construct_leaves(ar, size_data); // construct the leave nodes of the original bstree
    construct_inners(nextlevelnonleafkeys); // construct the inner nodes

    auto constructionStop =  high_resolution_clock::now();
    auto constructionDuration = duration_cast<milliseconds>(constructionStop - constructionStart);
    constructionTime = constructionDuration.count();
    cout << "Construction Time = " << constructionTime  << endl;

    //calculate the memory footprint
    memory_footprint(size_data);//calculate memory footprint

    //free auxiliary arrays
    free(nextlevelnonleafkeys); 
    free(ar);

    
    //read the workloads
    workloads_arr = read_work_load(workload_data, size_data, -1);

    //run the experiments
    auto sum_duration = 0;
	unsigned long long results = 0;
	uint64_t res = 0;
    
    auto start = high_resolution_clock::now();

    for(i=0;i<size_data;i++) {   
        res = execute_query(workloads_arr[i]);
        results ^= res;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    sum_duration = duration.count();
    uint32_t throughput =  ceil( (double)(size_data / (double)(sum_duration/1000.0))); //calculate the throughput

    cout << "BsTree: queries/second: "<< throughput << " results = "<< results  << " SIZE = "<< size_data <<endl;
	
    //free the workload vector and the the bstree
    workloads_arr.clear();
    delete_tree();
    
    return 0;
}
