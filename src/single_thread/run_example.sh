PATH="path-to/Bstree_sample_FB_dataset"
PATH_CONSTRUCTION="path-to/Bstree_sample_FB_dataset/construction_fb_50M.txt"
WORKLOADS=("read_fb_10M.txt" "write_fb_10M.txt" "read_write_fb_10M.txt" "range_write_fb_10M.txt" "range_write_fb_10M.txt")
WORKLOADNAMES=("WORKLOAD A (READ-ONLY)" "WORKLOAD B (WRITE-ONLY)" "WORKLOAD C (READ-WRITE)" "WORKLOAD D (RANGE-WRITE)" "WORKLOAD E (MIXED)")


for key in "${!WORKLOADS[@]}"
do
echo "########################### ${WORKLOADNAMES[$key]} ########################### "
echo "--------------------------- Bstree ---------------------------" 
./bstree $PATH_CONSTRUCTION $PATH/${WORKLOADS[$key]}

echo -e "\n--------------------------- Compact Bstree ---------------------------"
./compact_bstree $PATH_CONSTRUCTION $PATH/${WORKLOADS[$key]}
echo -e "\n--------------------------- Decision Bstree ---------------------------"
./decision_bstree $PATH_CONSTRUCTION $PATH/${WORKLOADS[$key]}
echo "############################################################################## "
echo -e "\n"
done
