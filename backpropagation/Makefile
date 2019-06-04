k_ = 10
folder = inputs

brew-path: 
	test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
	test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
	test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
	echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile

run: 
	python backpropagation.py backpropagation $(folder)/network.txt $(folder)/initial_weights.txt $(folder)/data_set.txt -v debug


	
