# DAMON-2025-BsTree
Code for  ** Bs-tree: A data-parallel B+-tree for main memory **


## Dependencies
- g++-13/gcc-13
- avx-512
- x86_64 Architecture

## Data
### Download
We have uploaded a sample of Facebook data in this Google Drive link ([link](https://drive.google.com/drive/folders/1e1SMt8VjAGEc-yS7CovtfutmzPktcBjE?usp=sharing)). The folder contains the following files:
- construction_fb_50M.txt: Index construction for 50 million entries.
- read_fb_10M.txt: 10 million entries (read-only).
- write_fb_10M.txt: 10 million entries (write-only).
- read_write_fb_10M.txt: 10 million entries (read and write operations).
- range_write_fb_10M.txt: 10 million entries (range-write operations).
- mixed_fb_10M.txt: 10 million entries (mixed read, write, and delete operations).

If you choose to use your own dataset, please ensure that the first line of each dataset specifies its size. We also have a mapping system where:
- 0 means **search**
- 1 means **write**
- 2 means **delete**
- 3 means **range**

So, the workloads dataset should be formatted as follows:
``` 
4
0 54135570494
2 72964226149
1 32326541453
3 7082951609 
```

## Compile & Run
Compile using ```make all``` or ```make <option>``` where `<option>` can be one of the following:

- bstree
- compact_bstree
- decision_bstree
- bstree_olc

Our code takes the construction dataset and the workload dataset as input arguments, in that order.
To run :
```
    ./bstree <construction dataset> <workload dataset>
    ./compact_bstree <construction dataset> <workload dataset>
    ./decision_bstree <construction dataset> <workload dataset>
    ./bstree_olc <construction dataset> <workload dataset>
```

## Indices
Our repository contains three approaches:
- Bs-tree, which executes the original Bs-tree.
- Compact Bs-tree, which executes the compact version of the Bs-tree.
- Decision Bs-tree, which analyzes the dataset to determine which of the above approaches to use.
- Bs-tree OLC, whicht executes the parallel version of bs-tree.


## Examples
We have also included a file named run_example.sh, which you can use to execute all our indices single-threaded on all workloads using the sample datasets. You only need to update the PATH and PATH_CONSTRUCTION variables.

To run:
```
./ run_example.sh
```
