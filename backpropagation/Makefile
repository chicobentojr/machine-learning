k_ = 10
folder = inputs
ex=1

brew-path: 
	test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
	test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
	test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
	echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile

backprop: 
	$(eval f := $(folder)/rede$(ex))
	python main.py backprop $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v debug

verification: 
	$(eval f := $(folder)/rede$(ex))
	python num_verification.py gradient_verification $(f)/network.txt $(f)/initial_weights.txt $(f)/data_set.txt -v debug


	
