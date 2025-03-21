
CC      = g++-13
CFLAGS  = -O3 -march=native

	
all: bstree compact_bstree decision_bstree 

bstree: 
	$(CC) $(CFLAGS) main_bstree.cpp -o bstree

compact_bstree: 
	$(CC) $(CFLAGS) main_compact_bstree.cpp -o compact_bstree

decision_bstree : 
	$(CC) $(CFLAGS) main_decision_bstree.cpp -o decision_bstree

clean:
	rm bstree
	rm compact_bstree
	rm decision_bstree