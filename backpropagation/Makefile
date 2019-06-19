k_ = 10
folder = inputs
ex=1
v=debug


backprop_1it: 
	$(eval f := $(folder)/rede$(ex))
	python main.py backprop $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -it 1 -v $(v)

backprop: 
	$(eval f := $(folder)/rede$(ex))
	python main.py backprop $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v $(v)

verification: 
	$(eval f := $(folder)/rede$(ex))
	python num_verification.py gradient_verification $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v $(v)

cross_wines: 
	python cross_validation_network.py execute datasets/wines.csv
