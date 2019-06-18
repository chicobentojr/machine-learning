k_ = 10
folder = inputs
ex=1
v=debug

brew-path: 
	test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
	test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
	test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
	echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile

run: 
	python backpropagation.py backpropagation $(folder)/network.txt $(folder)/initial_weights.txt $(folder)/data_set.txt -v $(v)

backprop: 
	$(eval f := $(folder)/rede$(ex))
	python backpropagation.py backpropagation $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v debug

verification: 
	$(eval f := $(folder)/rede$(ex))
	python num_verification.py numeric_verification $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v debug


	
