k_ = 10
folder = inputs
ex=1
v=debug
dataset=cancer


backprop_1it: 
	$(eval f := $(folder)/rede$(ex))
	python main.py backprop $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -it 1 -v $(v)

backprop: 
	$(eval f := $(folder)/rede$(ex))
	python main.py backprop $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v $(v)

verification: 
	$(eval f := $(folder)/rede$(ex))
	python num_verification.py gradient_verification $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v $(v)

ds: 
	python multiple_config_network.py execute datasets/$(dataset).csv -k 10 -p 50 -r 0.1 -r 0.25 -r 0.5 -arc 1 -arc 2 -arc 3 -arc 1,2 -arc 2,2 -arc 3,2 -m 100

dswines: 
	$(eval dataset:= wines)
	ds

dscancer: 
	$(eval dataset:= cancer)
	ds

dsiono: 
	$(eval dataset:= ionosphere)
	ds
	
